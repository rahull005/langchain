# Step 1: Import the Pydantic tools needed to define and validate a model.
# BaseModel creates the data model, EmailStr validates email addresses, and
# Field lets us add validation rules and documentation to a field.
from pydantic import BaseModel, EmailStr, Field

# Step 2: Import Optional so a field can either contain a value or None.
from typing import Optional


# Step 3: Define a Student schema by inheriting from Pydantic's BaseModel.
# Pydantic uses the type hints below to validate the input automatically.
class Student(BaseModel):
    # A string field with a default value if no name is supplied.
    name: str = "Rahul"

    # Optional integer: the age may be an integer or may be left as None.
    age: Optional[int] = None

    # Required field: EmailStr checks that the value has a valid email format.
    email: EmailStr

    # Float field with rules: CGPA must be greater than 0 and less than 10.
    # If omitted, the default CGPA is 5. The description documents the field.
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of the student",
    )


# Step 4: Prepare input data as a normal Python dictionary.
# The name and email are supplied; age is omitted, so it will become None.
new_student = {'name':'surya','email':'example@gmail.com','cgpa':5.5}

# Step 5: Unpack the dictionary and create a validated Student object.
# **new_student converts the dictionary keys into keyword arguments.
# Pydantic validates the email, data types, and CGPA rules here.
student = Student(**new_student)

# Step 6: Convert the Pydantic object into a regular dictionary.
# This is useful when sending or storing the data elsewhere.
student_dict = dict(student)

# Step 7: Access a validated field using normal object.attribute syntax.
print(student.name)

