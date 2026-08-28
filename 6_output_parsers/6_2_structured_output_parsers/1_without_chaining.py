from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
    )

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1",description="fact1 about the topic"),
    ResponseSchema(name="fact2",description="fact2 about the topic"),
    ResponseSchema(name="fact3",description="fact3 about the topic")
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


prompt = template.invoke({
    'topic':'Black hole'
})


response = model.invoke(prompt)
result = parser.parse(response.content)

print(result)