# Market Intel — Validated final release

The complete one-day Market Intel project is packaged in [Market_Intel_1D_Final_Validated.zip](./Market_Intel_1D_Final_Validated.zip).

Extract the ZIP, create a virtual environment, install requirements, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python validate_project.py
streamlit run app.py
```

See the included AUDIT_RESULTS.md and LAB_EVALUATION_GUIDE.md for methodology, benchmark results, limitations, and presentation notes. Predictive performance was weak and no reliable edge was established; the app uses a conservative HOLD gate when historical evidence is insufficient.
