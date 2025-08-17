from pydantic import BaseModel, Field
from typing import List

class CellStructure(BaseModel):
    value: str = Field(description="Cell content")
    row_headers: List[str] = Field(description="All row headers")
    col_headers: List[str] = Field(description="All column headers")

class TableStructure(BaseModel):
    cells: List[CellStructure]

def flatten_doctr_layout(doc_json, page_idx: int = 0):
    """
    Convert DocTR's nested blocks→lines→words structure into a flat list
    of positioned text elements for a single page.

    Returns
    -------
    List[dict]  # e.g. [{"text": "Col1", "x": 0.456, "y": 0.289, "conf": 0.996}, ...]
    """
    flat = []
    page = doc_json["pages"][page_idx]

    for block in page.get("blocks", []):
        for line in block.get("lines", []):
            for word in line.get("words", []):
                (x0, y0), (x1, y1) = word["geometry"]
                flat.append(
                    {
                        "text": word["value"],
                        "x": (x0 + x1) / 2,          # center-point for easier clustering
                        "y": (y0 + y1) / 2,
                        "width":  x1 - x0,
                        "height": y1 - y0,
                        "conf": word.get("confidence", None),
                    }
                )

    # optional: sort top-to-bottom, then left-to-right
    flat.sort(key=lambda el: (el["y"], el["x"]))
    return flat