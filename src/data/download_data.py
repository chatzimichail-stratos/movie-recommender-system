import os
import zipfile
import urllib.request
import ssl
import certifi
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-latest-small.zip")
    
    if not os.path.exists(zip_path):
        print("Κατέβασμα MovieLens dataset...")
        
        # Δημιουργούμε SSL context με τα certificates του certifi και το εγκαθιστούμε
        # ως προεπιλογή για το urllib, ώστε το urlretrieve να επαληθεύει σωστά το HTTPS
        # (λύνει το CERTIFICATE_VERIFY_FAILED στα Windows χωρίς να απενεργοποιεί τον έλεγχο ασφαλείας) 
        context = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

        print("Αποσυμπίεση...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    else:
        print("Το dataset υπάρχει ήδη στο data/raw!")

def process_initial_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    base_path = os.path.join(RAW_DIR, "ml-latest-small")
    
    # 1. Φόρτωση των CSV
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    links = pd.read_csv(os.path.join(base_path, "links.csv"))
    
    # 2. Join movies με links (για να έχουμε το tmdbId σε κάθε ταινία)
    movies = movies.merge(links[['movieId', 'tmdbId', 'imdbId']], on='movieId', how='left')
    
    # 3. Καθαρισμός: Το tmdbId μετά το left join έχει μερικά NaN, τα κάνουμε integer (nullable Int64)
    movies['tmdbId'] = movies['tmdbId'].astype('Int64')
    
    # 4. Αποθήκευση σε Parquet!
    movies.to_parquet(os.path.join(PROCESSED_DIR, "movies.parquet"), index=False)
    ratings.to_parquet(os.path.join(PROCESSED_DIR, "ratings.parquet"), index=False)

    # Το f κάνει το string ένα formatted string.
    # Μέσα στα {} βάζεις μεταβλητές ή εκφράσεις Python, και η τιμή τους μπαίνει στο κείμενο.
    print(f"Έτοιμα! Αποθηκεύτηκαν {len(movies)} ταινίες και {len(ratings)} βαθμολογίες σε Parquet.")  

# Εκτελείται ΜΟΝΟ όταν τρέχει το αρχείο απευθείας, όχι όταν γίνεται import
if __name__ == "__main__":
    download_and_extract()
    process_initial_data()