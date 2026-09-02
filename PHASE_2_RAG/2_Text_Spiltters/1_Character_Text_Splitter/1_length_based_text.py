from langchain_classic.text_splitter import CharacterTextSplitter


text = """
Find out which posts are a hit with Blogger’s built-in analytics. You’ll see where your audience is coming from and what they’re interested in. You can even connect your blog directly to Google Analytics for a more detailed look.

"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)


char_list = splitter.split_text(text)

print("Type of char_list",type(char_list))
print("Length of the char_list = ",len(char_list)) 

print("we got 3 docs based on the chunk size and overlap we provided to the splitter")

#Text_0
print("Text_0 :",char_list[0])


#Text-1
print("Text_1 :",char_list[1])


#Text-2
print("Text_2 :",char_list[2])
