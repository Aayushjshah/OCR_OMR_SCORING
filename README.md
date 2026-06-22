# OMR Evaluation Service

This service imports answer-key CSV files into JSON and scores OMR submissions from OCR output.

## Environment

```bash
export LIGHTON_OCR_BASE_URL="http://100.111.195.29:8000"
export LIGHTON_OCR_MODEL="lightonai/LightOnOCR-2-1B"
# Optional:
export LIGHTON_OCR_API_KEY=""
export OMR_KEY_DIR="./answer_keys"
export OMR_OUTPUT_DIR="./outputs"
export OMR_UPLOAD_DIR="./uploads"
export OMR_PDF_DPI="300"
```

For the bbox model, set:

```bash
export LIGHTON_OCR_BASE_URL="http://100.111.195.29:8002"
export LIGHTON_OCR_MODEL="lightonai/LightOnOCR-2-1B-bbox"
```

## Run

```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

## OCR Parsing

The scorer accepts saved LightOn chat-completions JSON, raw OCR text, PDFs, and images. For JSON, it reads `choices[0].message.content`.

The parser extracts:

- `Name:` / `Candidate Name:` / `Full Name:`
- `Email:` / `Email ID:` / `Email Address:`
- set labels such as `SET-1`, `Set: set1`, `Exam Set: set2`
- answers marked with filled OCR bubbles such as `A ●`, including HTML table cells from LightOn output

Unmarked answers are returned with `selected: null` and `status: "unmarked"`.

The plain text format below is also supported:

```text
Name: Candidate Name
Email: candidate@example.com
Set: set1
1: A
2: C
3: D
```

The OCR call itself is wired through the LightOn chat-completions endpoint.

Images and rendered PDF pages are preprocessed before OCR: EXIF orientation is normalized, blue/purple ink is darkened to black, and contrast/sharpness are increased. This improves detection of OMR bubbles filled with blue pen.

## Combined PDF

Use the Combined PDF upload when one PDF contains many student OMR sheets, one sheet per page. The service renders each page, scores it as one student, and returns a CSV.

Limits:

- Maximum combined PDF size: `40 MB`
- Maximum combined PDF pages: `250`
- Estimated processing time: `15 seconds/page`

Browser folder upload limits:

- Maximum files: `250`
- Maximum total upload size: `40 MB`

## Outputs

- Imported answer-key JSON files are stored in `answer_keys/`.
- OCR text and per-file scoring JSON are stored in `outputs/`.
- Batch scoring returns a CSV with `source,name,email,set,score,max_score,error`.
