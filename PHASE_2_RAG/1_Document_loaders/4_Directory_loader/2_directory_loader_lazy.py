# Import DirectoryLoader to load multiple documents from a directory
# Import PyPDFLoader to specifically handle PDF file parsing
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Initialize DirectoryLoader with configuration parameters
loader = DirectoryLoader(
    # Specify the directory path where PDF files are stored (using raw string to handle Windows backslashes)
    path=r'C:\Users\Sreenivas Bandaru\Desktop\langchain\PHASE_2_RAG\1_Document_loaders\4_Directory_loader\books',
    # Use glob pattern to filter files - only load files with .pdf extension
    glob='*.pdf',              # tells to load all the files with extension (.pdf)
    # Specify PyPDFLoader as the loader class to parse PDF content
    loader_cls=PyPDFLoader     # this is the loader_class
)

# Use lazy_load() to load documents on-demand (streaming mode) instead of loading all at once
# Returns a generator object that yields documents one at a time as they are requested
docs = loader.lazy_load()

# Print the length of the lazy_load generator (number of documents)
print(len(docs))

# Iterate through the lazy-loaded documents and print metadata from each
# Metadata typically includes: source file path, page number, and other document properties
for document in docs:
    print(document.metadata)

"""
LAZY LOADING EXPLANATION:
==========================

What is lazy_load()?
- lazy_load() is a memory-efficient way to load documents
- Instead of loading ALL documents into memory at once, it loads them one-by-one as needed
- Returns a generator/iterator object that yields documents sequentially

Benefits of lazy_load():
1. Memory Efficiency: Ideal when working with large numbers of files or very large documents
   - Only one document is loaded into memory at a time
   - Doesn't consume memory for all files upfront

2. Performance: Faster initial execution
   - Doesn't wait for all files to be loaded before processing starts
   - You can start processing documents immediately

3. Scalability: Can handle thousands of files without running out of memory
   - Perfect for production systems with large document collections

Alternative: load()
- load() loads ALL documents into memory at once
- Returns a list of all documents
- Use when you have a small number of files or need all data immediately

When to use lazy_load():
- Loading 100+ files
- Files are large (PDFs with many pages)
- Processing documents in batches
- Running on memory-constrained systems

When to use load():
- Loading a small number of files (< 10)
- Need all documents in memory for comparison
- File size is small
"""