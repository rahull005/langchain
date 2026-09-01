from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Can you explain {topic} in simple terms?"),
])

prompt = prompt_template.invoke({
    "domain": "medical",
    "topic": "diabetes"
})

print(prompt)

