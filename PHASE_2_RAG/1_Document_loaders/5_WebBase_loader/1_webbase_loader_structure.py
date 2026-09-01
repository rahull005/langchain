from langchain_community.document_loaders import WebBaseLoader



url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

docs = loader.load()


print("Type of docs",type(docs))

print("Length of the docs = ",len(docs)) #returns 1=>because iam retrinng single page


print("Page_Content = ",docs[0].page_content)
print("metadata = ",docs[0].metadata)

