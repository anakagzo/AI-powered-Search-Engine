import pypandoc
import tempfile
from fastapi import UploadFile

def convert_docx_to_markdown(file: UploadFile) -> str:
    """
    Converts a DOCX file to Markdown using Pandoc.
    Images are extracted to the /media directory.

    Args:
        file:  Uploaded file to be converted

    Returns:
        Markdown content as a string
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    output = pypandoc.convert_file(
        tmp_path,
        to="markdown",
        extra_args=["--extract-media=media"]
    )

    if not output or not output.strip():
        raise ValueError("Pandoc produced empty output")

    return output



