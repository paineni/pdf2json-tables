# Standard library imports
import os
import json
from typing import Optional, TypedDict

# Third-party imports
from IPython.display import Image, display

# Doctr (Document Understanding Transformer) imports
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Langchain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Local module imports
from utils import flatten_doctr_layout, TableStructure

# Define the Gemini-compatible endpoint
custom_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
output_folder = "./outputs"
# Load system prompt from file
with open("system_message.txt", "r", encoding="utf-8") as f:
    SYSTEM_MSG = f.read()

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("human", "{ocr_elements}")
])

# Initialize the language model
llm = ChatOpenAI(
    model="gemini-2.5-pro",
    openai_api_base=custom_api_base,
    openai_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    )

# Bind model to structured schema
structured_llm = llm.with_structured_output(TableStructure)


class State(TypedDict, total=False):
    pdf_path: str # Input
    doctr_output: dict # Raw OCR layout (from docTR)
    flattened_elements: list[dict] # Flattened version for prompt
    llm_output: TableStructure # Final structured result
    missing_texts: Optional[list[str]]  #  For feedback
    retry: bool # perform retry if True
    retry_count: int  #  To prevent infinite loops

def docTR_node(state: State) -> State:
    """
    OCR node: Runs docTR on the input PDF and saves the raw layout output to state.

    Expects:
        - state["pdf_path"]: string path to the input PDF file.

    Produces:
        - state["doctr_output"]: dict output from docTR.export()
    """
    # Load docTR model 
    model = ocr_predictor(pretrained=True)

    # Load the PDF file using docTR
    doc = DocumentFile.from_pdf(state["pdf_path"])

    # Perform OCR prediction
    result = model(doc)
    output_path = os.path.join(output_folder, "doctr_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.export(), f, indent=4)
    state["doctr_output"]=result.export()
    # Save the OCR output (as dict) into state
    return state 

def flatten_node(state: State) -> State:
    """
    Flattens the nested docTR OCR layout into a flat list of text elements with spatial coordinates.

    Expects:
        - state["doctr_output"]: dict from docTR.export()

    Produces:
        - state["flattened_elements"]: list of dicts like:
            {"text": "RPM", "x": 0.45, "y": 0.33}
    """
    doctr_output = state["doctr_output"]

    # Flatten the ocr oupt to a cleaner form
    flattened = flatten_doctr_layout(doctr_output)
    state["flattened_elements"]=flattened
    return state

def llm_node(state: State) -> State:
    """
    Uses the flattened OCR layout to reconstruct the table using a structured LLM prompt.

    Expects:
        - state["flattened_elements"]: list of dicts
        - state["missing_texts"]: list or string (for system message placeholder)

    Produces:
        - state["llm_output"]: parsed response from LLM based on schema
    """
    # 1. Serialize flattened layout
    flattened_ocr_elements = json.dumps(state["flattened_elements"], indent=4)

    # 2. Construct the prompt (inject missing_texts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human", "{ocr_elements}")
    ]).partial(missing_texts=state.get("missing_texts", "None"))

    chain = prompt | structured_llm

    # 3. Invoke the model
    response = chain.invoke({"ocr_elements": flattened_ocr_elements})
    # 4. Update state
    state["llm_output"]= response
    return state


def validate_node(state: State) -> State:
    """
    Validates whether the LLM output covers all OCR-extracted text values.

    Updates state with:
    - is_valid: bool — True if all OCR values are present in LLM output
    - missing_texts: list[str] — any missing values
    - retry: bool — True if retry is needed
    - retry_count: int — incremented if retrying
    """
    flattened = state.get("flattened_elements", [])
    llm_cells = state.get("llm_output", TableStructure(cells=[])).cells

    original_values = {el["text"] for el in flattened}
    predicted_values = {cell.value for cell in llm_cells}
    missing = sorted(original_values - predicted_values)

    state["missing_texts"] = missing
    state["retry_count"] = state.get("retry_count")

    # Retry only if missing and retry_count < 1
    state["retry"] = bool(missing) and state["retry_count"] < 1
    if state["retry"]:
        state["retry_count"] += 1
    return state

  
def validation_condition(state):
    """
    Checks if a retry is needed or if the process should end.
    Prints a warning if the process ends with missing values.
    """
    if state.get('retry', False):
        print(f"Retrying due to missing values: {state['missing_texts']}")
        return "llm"
    else:
        # Check for missing values on the final attempt
        missing_texts = state.get('missing_texts')
        if missing_texts:
            print("Warning: Process finished after retries, but some values are still missing.")
            print(f"Missing values: {missing_texts}")
        else:
            print("Validation passed successfully.")
        
        print("Proceeding to persist output.")
        return "persist"

    
def persist_output(state: State) -> State:
    """
    Final node that saves the structured LLM output to a JSON file,
    but only if the output passed validation.

    Updates:
    - Writes `llm_output` to `structured_output.json`
    """
    output_path = os.path.join(output_folder, "structured_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state["llm_output"].model_dump(), f, indent=4)

    print(f"Structured output saved to {output_path}")
    return state

if __name__ == "__main__":
    
    os.makedirs(output_folder, exist_ok=True)
    # Initialize a StateGraph using your custom State schema
    builder = StateGraph(State)

    # Register all processing nodes
    builder.add_node("ocr", docTR_node)             # Extract raw OCR layout using DocTR
    builder.add_node("flatten", flatten_node)  # Flatten nested OCR layout into structured list
    builder.add_node("llm", llm_node)          # Call LLM with flattened OCR + prompt + schema
    builder.add_node("validate", validate_node)  # Check if all OCR values are captured
    builder.add_node("persist", persist_output)  # Save structured output to JSON if valid


    # Define the execution flow between nodes
    builder.add_edge(START, "ocr")        # Start at the OCR node
    builder.add_edge("ocr", "flatten")    # OCR → flatten
    builder.add_edge("flatten", "llm")    # flatten → LLM
    builder.add_edge("llm", "validate")   # LLM → validate

    # The conditional edge correctly handles ALL routing from the "validate" node.
    builder.add_conditional_edges(
        "validate",            # Source node
        validation_condition,  # Function that returns the name of the next node
        {
            "llm": "llm",          
            "persist": "persist"  
        }
    )

    builder.add_edge("persist", END)      # persist is the final node

    # Compile the graph into a runnable object
    graph = builder.compile()

    try:
        # Generate PNG bytes
        mermaid_png = graph.get_graph().draw_mermaid_png()

        # Display in notebook
        display(Image(mermaid_png))

        # Save to file
        filename = os.path.join(output_folder, "graph_flow.png")
        with open(filename, 'wb') as f:
            f.write(mermaid_png)
            print(f"Mermaid diagram saved to: {filename}\n")

    except Exception as e:
        print("Failed to generate or display Mermaid PNG:", e)

    #Run the workflow with the initial input (PDF path)
    initial_state = State()
    initial_state["pdf_path"]= "Table-Example-R.pdf"
    initial_state["retry_count"]= 0
    initial_state["missing_texts"]= []


    final_state = graph.invoke(initial_state)
