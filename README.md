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

## How to Run

### Backend
```bash
cd backend
pip install fastapi uvicorn pdfplumber python-docx pillow pytesseract pdf2image
uvicorn main:app --reload
