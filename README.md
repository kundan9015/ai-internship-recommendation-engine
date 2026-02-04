# AI-Based Internship Recommendation Engine

This project is an AI-based internship recommendation system that parses resumes
(PDF, DOCX, Images) and recommends internships based on skill matching,
domain preference, and location relevance.

## Features
- Resume parsing using OCR and NLP
- Internship matching with scoring
- FastAPI backend
- HTML/CSS/JS frontend

## Tech Stack
- Python
- FastAPI
- OCR (Tesseract)
- HTML, CSS, JavaScript

<img width="1361" height="644" alt="Screenshot 2026-02-04 140251" src="https://github.com/user-attachments/assets/bce127d1-6ebd-4c96-ba76-3632c59e07a0" />
<img width="1338" height="642" alt="Screenshot 2026-02-04 140324" src="https://github.com/user-attachments/assets/3a27fa02-44c4-40ac-aaa9-6185adc6a5a1" />
<img width="1343" height="643" alt="Screenshot 2026-02-04 140342" src="https://github.com/user-attachments/assets/aec0507c-2be2-4b09-bcce-1a3d4078d4f2" /


## How to Run

### Backend
```bash
cd backend
pip install fastapi uvicorn pdfplumber python-docx pillow pytesseract pdf2image
uvicorn main:app --reload

>

