import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Page Configuration
st.set_page_config(
    page_title="GST Reconciliation Tool",
    layout="wide",
    page_icon="📊"
)

# Constants
REQUIRED_COLUMNS = [
    "GSTIN/UIN", "Supplier", "Invoice", "Date", 
    "Gross Amt", "Taxable", "IGST", "SGST", "CGST", "Type"
]

NUMERIC_COLUMNS = ["Gross Amt", "Taxable", "IGST", "SGST", "CGST"]



# Helper Functions
def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df


def validate_structure(df, filename):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"Error in {filename}: Missing columns: {', '.join(missing)}")
        return False
    return True


def preprocess_data(df):
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Clean Invoice Number
    if 'Invoice' in df.columns:
        df['Invoice'] = df['Invoice'].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')
        df['Invoice_Clean'] = df['Invoice'].str.strip().str.upper()

    # Clean GSTIN
    if 'GSTIN/UIN' in df.columns:
        df['GSTIN/UIN'] = df['GSTIN/UIN'].astype(str).replace('nan', '')
        df['GSTIN_Clean'] = df['GSTIN/UIN'].str.strip().str.upper()

    # Date parsing
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    return df


def get_target_periods(fy_string, period_type, selected_period):
    """Return list of (month, year) tuples."""
    start_year = int(fy_string.split('-')[0])
    end_year = int(fy_string.split('-')[1])
    
    target_dates = []
    period_label = ""

    if period_type == "Monthly":
        month_map = {
            "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9,
            "October": 10, "November": 11, "December": 12, "January": 1, "February": 2, "March": 3
        }
        m_num = month_map[selected_period]
        y_num = start_year if m_num >= 4 else end_year
        target_dates = [(m_num, y_num)]
        period_label = f"{selected_period} {y_num}"

    else: 
        if selected_period == "Q1 (Apr-Jun)":
            target_dates = [(4, start_year), (5, start_year), (6, start_year)]
        elif selected_period == "Q2 (Jul-Sep)":
            target_dates = [(7, start_year), (8, start_year), (9, start_year)]
        elif selected_period == "Q3 (Oct-Dec)":
            target_dates = [(10, start_year), (11, start_year), (12, start_year)]
        elif selected_period == "Q4 (Jan-Mar)":
            target_dates = [(1, end_year), (2, end_year), (3, end_year)]
        period_label = f"{selected_period} ({fy_string})"

    return target_dates, period_label




# Reconciliation Engine
def values_match_within_tolerance(val1, val2, tolerance):
    """Check if two values match within the given tolerance."""
    return abs(val1 - val2) <= tolerance


def run_reconciliation(df_2b, df_books, target_dates, tolerance=1):

    def is_in_period(df):
        valid = set(target_dates)
        return df['Date'].apply(lambda d: (d.month, d.year) in valid if pd.notnull(d) else False)
    
    def coalesce(row, columns):
        for col in columns:
            if col in row.index and pd.notnull(row[col]) and str(row[col]).strip() != "" and str(row[col]).lower() != "nan":
                return row[col]
        return ""

    # Filter
    df_books_current = df_books[is_in_period(df_books)].copy()
    df_2b_current = df_2b[is_in_period(df_2b)].copy()

    df_books_out = df_books[~is_in_period(df_books)].copy()
    df_books_out['Source'] = 'Books'

    df_2b_out = df_2b[~is_in_period(df_2b)].copy()
    df_2b_out['Source'] = 'GSTR-2B'

    df_out_of_period = pd.concat([df_books_out, df_2b_out], ignore_index=True)

    
    # STEP 1: Exact Match (Invoice + GSTIN)
    merged_step1 = pd.merge(
        df_2b_current,
        df_books_current,
        on=['GSTIN_Clean', 'Invoice_Clean'],
        how='outer',
        suffixes=('_2b', '_books'),
        indicator=True
    )

    exact_matches = merged_step1[merged_step1['_merge'] == 'both']
    leftover_2b = merged_step1[merged_step1['_merge'] == 'left_only']
    leftover_books = merged_step1[merged_step1['_merge'] == 'right_only']

    # STEP 2: Probable Match (GSTIN + Amount with Tolerance)
    
    # 2B Candidate
    cols_2b = {col: col.replace('_2b', '') for col in leftover_2b.columns if '_2b' in col}
    cols_2b.update({'GSTIN_Clean': 'GSTIN_Clean', 'Invoice_Clean': 'Invoice_Original_2B'})
    unique_cols_2b = list(dict.fromkeys(list(cols_2b.keys()) + ['GSTIN_Clean']))
    df_2b_candidate = leftover_2b[unique_cols_2b].rename(columns=cols_2b)

    # Books Candidate
    cols_books = {col: col.replace('_books', '') for col in leftover_books.columns if '_books' in col}
    cols_books.update({'GSTIN_Clean': 'GSTIN_Clean', 'Invoice_Clean': 'Invoice_Original_Books'})
    unique_cols_books = list(dict.fromkeys(list(cols_books.keys()) + ['GSTIN_Clean']))
    df_books_candidate = leftover_books[unique_cols_books].rename(columns=cols_books)

    # Apply tolerance-based matching
    probable_matches_list = []
    unmatched_2b_indices = set(df_2b_candidate.index)
    unmatched_books_indices = set(df_books_candidate.index)
    
    for idx_2b, row_2b in df_2b_candidate.iterrows():
        for idx_books, row_books in df_books_candidate.iterrows():
            if row_2b['GSTIN_Clean'] != row_books['GSTIN_Clean']:
                continue
            
            if (values_match_within_tolerance(row_2b['Taxable'], row_books['Taxable'], tolerance) and
                values_match_within_tolerance(row_2b['IGST'], row_books['IGST'], tolerance) and
                values_match_within_tolerance(row_2b['CGST'], row_books['CGST'], tolerance) and
                values_match_within_tolerance(row_2b['SGST'], row_books['SGST'], tolerance)):
                
                # Merge the row
                merged_row = {}
                
                # [FIX 1] Explicitly preserve GSTIN_Clean for the Results section
                merged_row['GSTIN_Clean'] = row_2b['GSTIN_Clean']

                for col in row_2b.index:
                    if col in row_books.index:
                        merged_row[f"{col}_2b"] = row_2b[col]
                        merged_row[f"{col}_books"] = row_books[col]
                    else:
                        merged_row[col] = row_2b[col]
                
                for col in row_books.index:
                    if col not in row_2b.index:
                        merged_row[col] = row_books[col]
                
                merged_row['_merge'] = 'both'
                probable_matches_list.append(merged_row)
                
                unmatched_2b_indices.discard(idx_2b)
                unmatched_books_indices.discard(idx_books)
                break
    
    # Create unmatched DataFrames
    only_2b_list = []
    for idx in unmatched_2b_indices:
        row = df_2b_candidate.loc[idx].to_dict()
        row['_merge'] = 'left_only'
        only_2b_list.append(row)
    
    only_books_list = []
    for idx in unmatched_books_indices:
        row = df_books_candidate.loc[idx].to_dict()
        row['_merge'] = 'right_only'
        only_books_list.append(row)
    
    merged_step2 = pd.DataFrame(probable_matches_list + only_2b_list + only_books_list)

    
    # STEP 3: Invoice Different, All Values Match
 
    if not merged_step2.empty and '_merge' in merged_step2.columns:
        probable_matches = merged_step2[merged_step2['_merge'] == 'both'].copy()
        
        if not probable_matches.empty:
            # Check if Gross Amt also matches
            invoice_diff_value_match = probable_matches[
                probable_matches.apply(
                    lambda row: values_match_within_tolerance(
                        row.get('Gross Amt_2b', 0), 
                        row.get('Gross Amt_books', 0), 
                        tolerance
                    ), axis=1
                )
            ].copy()
        else:
            invoice_diff_value_match = pd.DataFrame()
        
        probable_without_invoice_diff = merged_step2[
            ~merged_step2.index.isin(invoice_diff_value_match.index)
        ]
    else:
        invoice_diff_value_match = pd.DataFrame()
        probable_without_invoice_diff = merged_step2
        
    # Buiild Results
    
    results = []
    invoice_diff_results = []

    def safe_str(v):
        s = str(v)
        return "" if s == "nan" or s == "None" else s

    # A. Exact Matches
    for _, row in exact_matches.iterrows():
        remarks = []
        is_val_match = True

        for col in NUMERIC_COLUMNS:
            val_2b = row.get(f"{col}_2b", 0)
            val_books = row.get(f"{col}_books", 0)
            if not values_match_within_tolerance(val_2b, val_books, tolerance):
                is_val_match = False
                diff = abs(val_2b - val_books)
                remarks.append(f"{col} diff: ₹{diff:.2f}")

        status = "Matched" if is_val_match else "Mismatch in Value"

        results.append({
            'Status': status,
            'Remarks': ", ".join(remarks) if remarks else f"Matched within ±₹{tolerance}",
            'GSTIN': safe_str(row.get('GSTIN/UIN_2b')),
            'Supplier': safe_str(row.get('Supplier_2b')),
            'Invoice': safe_str(row.get('Invoice_2b')),
            'Date': row.get('Date_2b'),
            'Taxable_2B': row.get('Taxable_2b', 0),
            'Taxable_Books': row.get('Taxable_books', 0),
            'Tax_2B': row.get('IGST_2b', 0) + row.get('CGST_2b', 0) + row.get('SGST_2b', 0),
            'Tax_Books': row.get('IGST_books', 0) + row.get('CGST_books', 0) + row.get('SGST_books', 0)
        })

    # B. Invoice Different but Values Match
    for _, row in invoice_diff_value_match.iterrows():
        # [FIX 2] Use coalesce to find data in either _2b, _books, or plain columns
        invoice_diff_results.append({
            'GSTIN': safe_str(coalesce(row, ['GSTIN_Clean', 'GSTIN_Clean_2b', 'GSTIN_Clean_books'])),
            'Supplier': safe_str(coalesce(row, ['Supplier_2b', 'Supplier_books', 'Supplier'])),
            'Invoice_2B': safe_str(coalesce(row, ['Invoice_Original_2B', 'Invoice_Original_2B_2b'])),
            'Invoice_Books': safe_str(coalesce(row, ['Invoice_Original_Books', 'Invoice_Original_Books_books'])),
            'Date_2B': coalesce(row, ['Date_2b', 'Date']),
            'Date_Books': coalesce(row, ['Date_books', 'Date']),
            'Gross_Amt': row.get('Gross Amt_2b', 0),
            'Taxable': row.get('Taxable_2b', row.get('Taxable', 0)),
            'IGST': row.get('IGST_2b', row.get('IGST', 0)),
            'CGST': row.get('CGST_2b', row.get('CGST', 0)),
            'SGST': row.get('SGST_2b', row.get('SGST', 0)),
            'Total_Tax': (row.get('IGST_2b', row.get('IGST', 0)) + 
                          row.get('CGST_2b', row.get('CGST', 0)) + 
                          row.get('SGST_2b', row.get('SGST', 0)))
        })

    # C. Probable Matches
    if not probable_without_invoice_diff.empty and '_merge' in probable_without_invoice_diff.columns:
        probable = probable_without_invoice_diff[probable_without_invoice_diff['_merge'] == 'both']
        for _, row in probable.iterrows():
            results.append({
                'Status': "Probable Match",
                'Remarks': f"Invoice mismatch: 2B[{safe_str(row.get('Invoice_Original_2B', ''))}] vs Books[{safe_str(row.get('Invoice_Original_Books', ''))}]",
                'GSTIN': safe_str(coalesce(row, ['GSTIN_Clean', 'GSTIN_Clean_2b', 'GSTIN_Clean_books'])),
                'Supplier': safe_str(coalesce(row, ['Supplier_2b', 'Supplier'])),
                'Invoice': f"{safe_str(row.get('Invoice_Original_2B', ''))} / {safe_str(row.get('Invoice_Original_Books', ''))}",
                'Date': coalesce(row, ['Date_2b', 'Date']),
                'Taxable_2B': row.get('Taxable_2b', row.get('Taxable', 0)),
                'Taxable_Books': row.get('Taxable_books', row.get('Taxable', 0)),
                'Tax_2B': (row.get('IGST_2b', row.get('IGST', 0)) + 
                           row.get('CGST_2b', row.get('CGST', 0)) + 
                           row.get('SGST_2b', row.get('SGST', 0))),
                'Tax_Books': (row.get('IGST_books', row.get('IGST', 0)) + 
                             row.get('CGST_books', row.get('CGST', 0)) + 
                             row.get('SGST_books', row.get('SGST', 0)))
            })

        # D. Only in 2B
        only_2b = probable_without_invoice_diff[probable_without_invoice_diff['_merge'] == 'left_only']
        for _, row in only_2b.iterrows():
            results.append({
                'Status': "Only in 2B",
                'Remarks': "Missing in Books",
                'GSTIN': safe_str(coalesce(row, ['GSTIN_Clean', 'GSTIN_Clean_2b'])),
                'Supplier': safe_str(coalesce(row, ['Supplier', 'Supplier_2b'])),
                'Invoice': safe_str(coalesce(row, ['Invoice_Original_2B', 'Invoice'])),
                'Date': coalesce(row, ['Date', 'Date_2b']),
                'Taxable_2B': row.get('Taxable', row.get('Taxable_2b', 0)),
                'Taxable_Books': 0,
                'Tax_2B': (row.get('IGST', row.get('IGST_2b', 0)) + 
                           row.get('CGST', row.get('CGST_2b', 0)) + 
                           row.get('SGST', row.get('SGST_2b', 0))),
                'Tax_Books': 0
            })

        # E. Only in Books
        only_books = probable_without_invoice_diff[probable_without_invoice_diff['_merge'] == 'right_only']
        for _, row in only_books.iterrows():
            results.append({
                'Status': "Only in Books",
                'Remarks': "Missing in 2B",
                'GSTIN': safe_str(coalesce(row, ['GSTIN_Clean', 'GSTIN_Clean_books'])),
                'Supplier': safe_str(coalesce(row, ['Supplier', 'Supplier_books'])),
                'Invoice': safe_str(coalesce(row, ['Invoice_Original_Books', 'Invoice'])),
                'Date': coalesce(row, ['Date', 'Date_books']),
                'Taxable_2B': 0,
                'Taxable_Books': row.get('Taxable', row.get('Taxable_books', 0)),
                'Tax_2B': 0,
                'Tax_Books': (row.get('IGST', row.get('IGST_books', 0)) + 
                             row.get('CGST', row.get('CGST_books', 0)) + 
                             row.get('SGST', row.get('SGST_books', 0)))
            })

    df_results = pd.DataFrame(results)
    df_invoice_diff = pd.DataFrame(invoice_diff_results)

    if not df_results.empty:
        df_results['GSTIN'] = df_results['GSTIN'].astype(str)
        df_results['Invoice'] = df_results['Invoice'].astype(str)
        df_results['Supplier'] = df_results['Supplier'].astype(str)
        df_results['Date'] = pd.to_datetime(df_results['Date'], errors='coerce')
        df_results = df_results.sort_values(by='Date')

    if not df_invoice_diff.empty:
        df_invoice_diff['Date_2B'] = pd.to_datetime(df_invoice_diff['Date_2B'], errors='coerce')
        df_invoice_diff['Date_Books'] = pd.to_datetime(df_invoice_diff['Date_Books'], errors='coerce')
        df_invoice_diff = df_invoice_diff.sort_values(by='Date_2B')

    return df_results, df_out_of_period, df_invoice_diff




# Pdf Generator

def generate_pdf_summary(metrics, period_label):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "GST Reconciliation Summary Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Period: {period_label}")
    c.line(50, height - 80, width - 50, height - 80)

    y = height - 120
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Overview:")
    y -= 30

    c.setFont("Helvetica", 12)
    for label, value in [
        ("Fully Matched Records", metrics['matched']),
        ("Value Mismatches", metrics['mismatch']),
        ("Probable Matches", metrics['probable']),
        ("Invoice Diff (Values Match)", metrics['invoice_diff']),
        ("Only in 2B", metrics['only_2b']),
        ("Only in Books", metrics['only_books']),
        ("Out of Period Records", metrics['out_period'])
    ]:
        c.drawString(70, y, f"{label}:")
        c.drawString(350, y, str(value))
        y -= 25

    c.save()
    buffer.seek(0)
    return buffer




# Streamlit UI


st.title("📊 GST Reconciliation Tool")
st.markdown("Compare **GSTR-2B** with your **Purchase Register**.")

with st.sidebar:
    st.header("Configuration")

    current_year = datetime.now().year
    fy_options = [f"{y}-{y+1}" for y in range(current_year - 3, current_year + 2)]
    selected_fy = st.selectbox("Financial Year", fy_options, index=2)

    period_type = st.radio("Period Type", ["Monthly", "Quarterly"])
    
    st.markdown("---")
    st.subheader("Tolerance Settings")
    tolerance = st.selectbox(
        "Amount Tolerance (₹)",
        options=[0, 1, 5, 10, 20, 50],
        index=1,
        help="Allow matching when amounts differ by up to this value"
    )
    st.caption(f"Current: ±₹{tolerance}")

    if period_type == "Monthly":
        month_options = [
            "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "January", "February", "March"
        ]
        selected_period_val = st.selectbox("Select Month", month_options)
    else:
        quarter_options = [
            "Q1 (Apr-Jun)", "Q2 (Jul-Sep)", "Q3 (Oct-Dec)", "Q4 (Jan-Mar)"
        ]
        selected_period_val = st.selectbox("Select Quarter", quarter_options)

    st.markdown("---")
    file_2b = st.file_uploader("Upload GSTR-2B", type=['xlsx'])
    file_books = st.file_uploader("Upload Books", type=['xlsx'])

# Run Process

if file_2b and file_books:
    try:
        df_2b = pd.read_excel(file_2b)
        df_books = pd.read_excel(file_books)

        df_2b = normalize_columns(df_2b)
        df_books = normalize_columns(df_books)

        if validate_structure(df_2b, "GSTR-2B") and validate_structure(df_books, "Books"):

            df_2b_clean = preprocess_data(df_2b.copy())
            df_books_clean = preprocess_data(df_books.copy())

            target_dates, period_label = get_target_periods(
                selected_fy, period_type, selected_period_val
            )

            recon_results, df_out_of_period, df_invoice_diff = run_reconciliation(
                df_2b_clean, df_books_clean, target_dates, tolerance
            )

            matched = len(recon_results[recon_results['Status'] == "Matched"])
            mismatch = len(recon_results[recon_results['Status'] == "Mismatch in Value"])
            probable = len(recon_results[recon_results['Status'] == "Probable Match"])
            only_2b = len(recon_results[recon_results['Status'] == "Only in 2B"])
            only_books = len(recon_results[recon_results['Status'] == "Only in Books"])
            out_period = len(df_out_of_period)
            invoice_diff = len(df_invoice_diff)

            st.subheader(f"Summary – {period_label}")
            st.caption(f"Amount Tolerance: ±₹{tolerance}")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Matched", matched)
            c2.metric("Value Mismatch", mismatch)
            c3.metric("Probable", probable)
            c4.metric("Invoice Diff", invoice_diff, help="Values match but invoice numbers differ")
            c5.metric("Only in 2B", only_2b)
            c6.metric("Only in Books", only_books)

            if out_period > 0:
                st.warning(f"{out_period} records are outside selected period.")
            
            if invoice_diff > 0:
                st.info(f"⚠️ {invoice_diff} records found where all values match but invoice numbers differ. Check 'Invoice Number Mismatch' tab.")

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Matched", "Value Mismatch", "Probable", "Invoice Number Mismatch", "Only in 2B", "Only in Books", "Out of Period"]
            )

            with tab1: st.dataframe(recon_results[recon_results['Status'] == "Matched"])
            with tab2: st.dataframe(recon_results[recon_results['Status'] == "Mismatch in Value"])
            with tab3: st.dataframe(recon_results[recon_results['Status'] == "Probable Match"])
            with tab4: 
                if invoice_diff == 0:
                    st.success("No records with invoice number mismatches where values match.")
                else:
                    st.warning("⚠️ These entries have matching GSTIN and all tax values, but different invoice numbers. Please verify which invoice number is correct.")
                    st.dataframe(df_invoice_diff)
            with tab5: st.dataframe(recon_results[recon_results['Status'] == "Only in 2B"])
            with tab6: st.dataframe(recon_results[recon_results['Status'] == "Only in Books"])
            with tab7:
                if out_period == 0:
                    st.info("No out-of-period records.")
                else:
                    st.dataframe(df_out_of_period)

            
            # Excel Export
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                
                sheet_mapping = {
                    "Matched": "Matched",
                    "Mismatch in Value": "Value Mismatch",
                    "Probable Match": "Probable",
                    "Only in 2B": "Only in 2B",
                    "Only in Books": "Only in Books"
                }

                for status_key, sheet_name in sheet_mapping.items():
                    
                    subset = recon_results[recon_results['Status'] == status_key]
                    
                   
                    if not subset.empty:
                        
                        subset_clean = subset.drop(columns=['Status'], errors='ignore')
                        subset_clean.to_excel(writer, index=False, sheet_name=sheet_name)
                
               
                if not df_invoice_diff.empty:
                    df_invoice_diff.to_excel(writer, index=False, sheet_name="Invoice Number Mismatch")
                
                if not df_out_of_period.empty:
                    df_out_of_period.to_excel(writer, index=False, sheet_name="Out of Period")

            st.download_button(
                label="📥 Download Detailed Excel Report",
                data=output_excel.getvalue(),
                file_name=f"Reconciliation_{period_label.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            metrics = {
                "matched": matched,
                "mismatch": mismatch,
                "probable": probable,
                "invoice_diff": invoice_diff,
                "only_2b": only_2b,
                "only_books": only_books,
                "out_period": out_period
            }

            pdf_bytes = generate_pdf_summary(metrics, period_label)

            st.download_button(
                label="📄 Download PDF Summary",
                data=pdf_bytes,
                file_name=f"Summary_{period_label.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload both files to begin reconciliation.")