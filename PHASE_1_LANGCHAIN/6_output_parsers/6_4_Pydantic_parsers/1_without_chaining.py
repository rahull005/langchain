from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
    )

model = ChatHuggingFace(llm=llm)


class Student(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")
    study: str = Field(description="The person's field of study")


parser = PydanticOutputParser(pydantic_object=Student)

template = PromptTemplate(
    template=(
        "Create an imaginary student who is from or associated with {place}. "
        "Provide the student's name, age, and field of study.\n"
        "{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


prompt = template.invoke({
    'place':'Hyderabad'
})


response = model.invoke(prompt)
result = parser.parse(response.content)

print(result)