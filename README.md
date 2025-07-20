
# Legal Text Classifier (TF-IDF + Naive Bayes)

This project classifies legal queries into predefined categories using machine learning.

## Supported Labels
- `cancellation`
- `refund`
- `policy`

## How It Works
- Text data is vectorized using **TF-IDF**
- A **Multinomial Naive Bayes** classifier is trained on labeled queries
- Evaluation is done using **accuracy score** and a **confusion matrix**

## Project Structure
.
├── data/
│ └── legal_queries.csv
├── notebook/
│ └── legal_csv_classifier.ipynb
├── scripts/
│ └── evaluate_classifier.py
├── requirements.txt
└── README.md


## How to Run

1. Create a virtual environment:

```bash
python -m venv .venv
```
2. Activate it:
Windows:
```
.venv\Scripts\activate
```
macOS/Linux:
```
source .venv/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the classifier:
```
python scripts/evaluate_classifier.py
```

## Example Output
```
✅ Accuracy: 1.0
📊 Confusion Matrix:
[[1 0 0]
 [0 2 0]
 [0 0 1]]
```
## Author
```
Raghuramreddy Thirumalareddy
```
