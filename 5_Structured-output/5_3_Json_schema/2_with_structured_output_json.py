# Step 1: The model's output must follow the structure defined by json_schema.
# This example uses a normal Python dictionary to describe the JSON Schema.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
import os

# Step 2: Load environment variables from .env, including the Hugging Face token.
load_dotenv()

# Step 3: Create an endpoint for the selected open-source Hugging Face model.
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

# Step 4: Wrap the endpoint as a chat model.
model = ChatHuggingFace(llm=llm)


# Step 5: Define the JSON Schema that the model's response must follow.
# Each property describes the expected field type and its purpose.
json_schema = {
  # The schema title identifies this structure.
  "title": "Review",

  # The complete response must be a JSON object, similar to a Python dict.
  "type": "object",
  "properties": {
    # key_themes must be a list containing only strings.
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },

    # summary must contain one string describing the review briefly.
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },

    # sentiment must be one of the values listed in enum.
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },

    # pros and cons may be a list of strings or null when no value is found.
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },

    # name may be a string or null when the reviewer is not identified.
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },

  # These fields must appear in the model's response.
  # Fields not listed here are optional according to JSON Schema.
  "required": ["key_themes", "summary", "sentiment"]
}


# Step 6: Create a runnable that asks the chat model to follow json_schema.
# The result is returned as a Python dictionary containing JSON-compatible data.
structured_model = model.with_structured_output(json_schema)


# Step 7: Prepare the review text that will be analyzed.
review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""


# Step 8: Invoke the structured runnable with the review text.
# The model extracts the review and returns data matching the schema.
response = structured_model.invoke(review)

# Step 9: Display the structured response.
print(response)





