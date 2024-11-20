import random
import string
import os
from datetime import datetime
import shutil


# Example usage
inputFilenameDict = {
    "BankStatements": ['bank_statement', 'statement_bank', 'bankStatements'],  # Variations for bank statements
    "Invoice": ['invoice', 'NewInvoice'],  # Variations for invoices
    "Payslips": ['payslip', 'pay_slip', 'salary_slip', 'payroll'],
    "TaxDocuments": ['tax_return', 'tax_statement', 'tax_doc', 'taxform'],
    "Contracts": ['contract', 'agreement', 'employment_contract', 'lease_agreement'],
    "UtilityBills": ['utility_bill', 'electricity_bill', 'water_bill', 'gas_bill', 'phone_bill'],
    "Receipts": ['receipt', 'purchase_receipt', 'sales_receipt', 'transaction_slip'],
    "InsuranceDocuments": ['insurance_policy', 'insurance_doc', 'policy_statement'],
    "IDProofs": ['passport', 'id_card', 'drivers_license', 'national_id'],
    "EducationalCertificates": ['degree_certificate', 'diploma', 'transcript', 'education_record'],
    "MedicalRecords": ['medical_report', 'health_record', 'doctor_prescription', 'lab_result'],
    "LoanDocuments": ['loan_agreement', 'loan_statement', 'mortgage_doc'],
    "PurchaseOrders": ['purchase_order', 'po', 'order_form', 'supplier_invoice'],
    "BankCheques": ['bank_cheque', 'cheque_image', 'check'],
    "MeetingMinutes": ['meeting_minutes', 'meeting_notes', 'minutes_of_meeting'],
    "LegalDocuments": ['court_order', 'legal_notice', 'affidavit', 'power_of_attorney'],
    "ShippingDocuments": ['bill_of_lading', 'packing_list', 'shipping_invoice'],
    "JobApplications": ['resume', 'cv', 'cover_letter', 'job_application'],
    "Miscellaneous": ['misc_doc', 'random_file', 'unknown', 'miscellaneous_document']
}





MaxVariationNumber = 100000





total_variations = sum(len(variations) for variations in inputFilenameDict.values())


def create_result_folder(folder_path = 'genResult'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, folder_path)
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Delete the existing folder and its contents
        print("removed: ", folder_name)
        shutil.rmtree(folder_name)
    # Create the folder
    os.makedirs(folder_name)
    return folder_name



def create_filename_variations(input_name,maxMistakes = 5):
    # Function to generate random capitalization variations
    def capitalize_variation(name):
        return ''.join(random.choice([c.upper(), c.lower()]) for c in name)

    # Function to introduce random spelling mistakes
    def introduce_mistake(word):
        word_list = list(word)
        #mistake_type = random.choice(['insert', 'delete', 'replace', 'transpose', 'none'])
        numOfMistake = random.randrange(0, min( len(word), maxMistakes ) )

        for i in range(numOfMistake):

            mistake_type = random.choice(['insert', 'delete', 'replace', 'transpose', 'none'])
            if mistake_type == 'insert' and len(word_list) > 1:
                # Insert a random letter in a random position
                pos = random.randint(0, len(word_list))
                letter = random.choice(string.ascii_lowercase)
                word_list.insert(pos, letter)
            elif mistake_type == 'delete' and len(word_list) > 1:
                # Delete a random letter
                pos = random.randint(0, len(word_list) - 1)
                word_list.pop(pos)
            elif mistake_type == 'replace' and len(word_list) > 1:
                # Replace a random letter with another letter
                pos = random.randint(0, len(word_list) - 1)
                letter = random.choice(string.ascii_lowercase)
                word_list[pos] = letter
            elif mistake_type == 'transpose' and len(word_list) > 2:
                # Swap two random adjacent letters
                pos = random.randint(0, len(word_list) - 2)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]

        return ''.join(word_list)

    # Function to add a timestamp variation (e.g., hsbc_bank_statement_20220123)
    def add_date_variation(name):
        date_str = datetime.now().strftime("%Y%m%d")  # Current date in YYYYMMDD format
        return f"{name}_{date_str}"
    
    def add_random_characters(input_name):
        return f"{input_name}_{random.choice(string.ascii_lowercase)}{random.randint(100, 999)}"
    
    def camel_case_variation(name):
        words = name.split("_")
        return ''.join([words[0].lower()] + [word.capitalize() for word in words[1:]])
    
    def applyALL(name):

        name = random.choice([name, capitalize_variation(name)])
        name = random.choice([name, introduce_mistake(name)])
        name = random.choice([name, add_date_variation(name)])
        name = random.choice([name, add_random_characters(name)])
        name = random.choice([name, camel_case_variation(name)])
        
        return name


    # Split the input into words (in case underscores exist)
    words = input_name.split("_")

    # Generate variations
    variations = []

    # 1. Capitalized version (e.g., "bankStatement")
    variations.append(camel_case_variation(input_name))

    # 2. Random capitalization variation (e.g., "Bank_Statements")
    variations.append(capitalize_variation(input_name))

    # 3. Add Date Variation (e.g., "hsbc_bank_statement_20220123")
    variations.append(add_date_variation(input_name))

    # 4. Mistakes in spelling (random letter variations in each word)
    mistyped_words = [introduce_mistake(word) for word in words]
    mistyped_filename = "_".join(mistyped_words)
    variations.append(mistyped_filename)

    # 5. Remove underscores to create a compact version (e.g., "bankstatements")
    variations.append(input_name.replace("_", ""))

    # 6. Add random characters (e.g., "bank_statements_a123")
    variations.append(add_random_characters(input_name))

    #7. Apply all randomely
    variations.append(applyALL(input_name))

    return variations



# Save variations to a specified folder
def save_variations_to_file(folder_name, input_name, variations, parentFolder = 'genResult'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, parentFolder, folder_name)

    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate the base filename
    base_filename = os.path.join(folder_path, input_name)
    filename = base_filename

    # Check if the file already exists and increment the filename
    counter = 1
    while os.path.exists(f"{filename}.txt"):
        filename = f"{base_filename}{counter}"
        counter += 1

    # Write variations to the file
    with open(f"{filename}.txt", 'w') as f:
        for variation in variations:
            f.write(variation + '\n')

    print(f"File saved as {filename}.txt")

# Run function to generate and save variations
def run(input_name, folder_name, max_variations=100):
    variations = []
    for _ in range(int(max_variations / 7)):
        var = create_filename_variations(input_name)
        variations.extend(var)

    # Remove duplicates
    variations = list(set(variations))

    # Save variations to a file in the specified folder
    save_variations_to_file(folder_name, input_name, variations)
    print("Number of variations:", len(variations))

# Main loop to process each entry in inputFilenameDict
create_result_folder()
for folder_name, filenames in inputFilenameDict.items():
    for input_name in filenames:
        run(input_name, folder_name, max_variations = MaxVariationNumber)





