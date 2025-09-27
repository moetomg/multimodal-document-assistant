import os
from langchain_unstructured import UnstructuredLoader
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
from typing import List, Dict, Any

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text, tables, formulas, and images from a document using Unstructured and PyMuPDF.
    Returns a list of content chunks with type, content, page, and source info.
    """
    print(f"--- Processing document: {os.path.basename(file_path)} ---")

    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        mode="elements",
        infer_table_structure=True,
        languages=["chi_sim", "eng"],
    )

    results = []
    try:
        elements = loader.load()
        for element in elements:
            if element.metadata.get('category') != "Image":
                content = element.page_content
                if content.strip():
                    element_type = element.metadata.get('category', 'Unknown')
                    if element_type == "Title":
                        formatted_content = f"# {content}"
                    elif element_type in ["Header", "SubTitle"]:
                        formatted_content = f"## {content}"
                    elif element_type == "ListItem":
                        formatted_content = f"* {content}"
                    elif element_type == "Table":
                        formatted_content = f"\n{content}\n"
                    elif element_type == "Formula":
                        bbox = element.metadata.get('bbox')
                        if bbox:
                            try:
                                page = pdf_doc.load_page(page_num - 1)
                                clip_rect = fitz.Rect(bbox)
                                pix = page.get_pixmap(clip=clip_rect, dpi=150)
                                pil_image = Image.open(io.BytesIO(pix.tobytes("png")))
                                results.append({
                                    "type": "image_formula",
                                    "content": pil_image,
                                    "page": element.metadata.get('page_number', 1),
                                    "source": os.path.basename(file_path)
                                })
                            except Exception as e:
                                print(f"WARNING: Could not clip formula image. Error: {e}")
                    else:
                        formatted_content = content
                    results.append({
                        "type": "text",
                        "content": formatted_content,
                        "page": element.metadata.get('page_number', 1),
                        "source": os.path.basename(file_path)
                    })
    except Exception as e:
        print(f"An error occurred during Unstructured processing: {e}")

    is_pdf = file_path.lower().endswith('.pdf')
    if is_pdf:
        try:
            pdf_doc = fitz.open(file_path)
            print("Processing PDF for images and scanned pages...")
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                embedded_images = page.get_images(full=True)
                if embedded_images:
                    for _, img in enumerate(embedded_images):
                        xref = img[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        results.append({
                            "type": "image",
                            "content": pil_image,
                            "page": page_num + 1,
                            "source": os.path.basename(file_path)
                        })
                else:
                    pix = page.get_pixmap(dpi=150)
                    pil_image = Image.open(io.BytesIO(pix.tobytes("png")))
                    results.append({
                        "type": "image",
                        "content": pil_image,
                        "page": page_num + 1,
                        "source": os.path.basename(file_path)
                    })
            pdf_doc.close()
        except Exception as e:
            print(f"An error occurred during PDF image processing: {e}")
    else:
        print("Processing non-PDF file for images from Unstructured metadata...")
        if 'elements' in locals():
            for element in elements:
                if element.metadata.get('category') == "Image":
                    image_bytes = element.metadata.get('image_bytes')
                    if image_bytes:
                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            results.append({
                                "type": "image",
                                "content": pil_image,
                                "page": element.metadata.get('page_number', 1),
                                "source": os.path.basename(file_path)
                            })
                        except:
                            pass
    print(f"--- Document processing finished. Total chunks: {len(results)} ---")
    return results

if __name__ == '__main__':
    test_folder = "files"
    test_file_name = "literature1.pdf"
    test_file_path = os.path.join(test_folder, test_file_name)

    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
    else:
        print(f"--- Running standalone test for document_processor.py ---")
        print(f"--- Processing file: {test_file_name} ---")
        processed_data = process_document(test_file_path)
        text_count = sum(1 for item in processed_data if item['type'] == 'text')
        image_count = sum(1 for item in processed_data if item['type'] == 'image')
        print(f"\n--- Test Results for {test_file_name} ---")
        print(f"Total extracted chunks: {len(processed_data)}")
        print(f"Text chunks: {text_count}, Image chunks: {image_count}")