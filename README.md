# 📊 GST Reconciliation Tool

A full-stack ready Streamlit application designed to automate the reconciliation process between **GSTR-2B** (Government Portal Data) and the **Purchase Register** (Books of Accounts). This tool identifies exact matches, probable matches (invoice number typos), and mismatches in tax values.

## 🚀 Key Features

- **Automated Matching:**
  - **Exact Match:** Matches records where GSTIN and Invoice Number are identical.
  - **Probable Match:** Identifies records where GSTIN and Tax Values match, but Invoice Numbers differ (e.g., "INV/001" vs "INV-1").
  - **Value Mismatch:** Flags records where Invoice/GSTIN match but tax amounts differ beyond a specified tolerance.
- **Flexible Configuration:** Support for Monthly/Quarterly reconciliation across multiple financial years.
- **Tolerance Threshold:** User-defined tolerance (e.g., ±₹1.00) for rounding differences.
- **Reporting:** Generates detailed Excel reports and PDF summaries.

## 📋 Input File Requirements

For the tool to function correctly, both the **GSTR-2B** and **Purchase Register** Excel files must contain the exact column headers listed below.

**Required Columns (Case-Sensitive):**

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `GSTIN/UIN` | String | The Supplier's GSTIN. |
| `Supplier` | String | Name of the Supplier. |
| `Invoice` | String | Invoice Number. |
| `Date` | Date/String | Invoice Date (DD-MM-YYYY). |
| `Gross Amt` | Numeric | Total Invoice Value. |
| `Taxable` | Numeric | Taxable Value. |
| `IGST` | Numeric | Integrated Goods and Services Tax. |
| `SGST` | Numeric | State Goods and Services Tax. |
| `CGST` | Numeric | Central Goods and Services Tax. |
| `Type` | String | Invoice Type (e.g., B2B). |

> **Note:** The tool includes a pre-processing step that normalizes column names by stripping whitespace, but the spelling must match exactly.

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python installed (version 3.8 or higher).

### 2. Install Dependencies
Run the following command to install the required libraries:

```bash
pip install streamlit pandas numpy reportlab xlsxwriter openpyxl
```
### 2. Run the Application
streamlit run filter3.py
