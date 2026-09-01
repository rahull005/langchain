from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str


person : Person = {
    "name": "John Doe",
    "age": 30,                              #here if you pass, string also it will not throw an error, but it will be a type mismatch.
    "email": "john.doe@example.com"
}


print(person)

#here the TypedDict class is used to define a dictionary type with specific keys and their corresponding value types. In this case, the Person TypedDict has three keys: name (str), age (int), and email (str). The person variable is then created as an instance of the Person TypedDict, and it is printed to the console.

#and the field type is not mandatory, like the type is integer but the value is string, it will not throw an error, but it will be a type mismatch.
# here below the age wantedly passed as string to show the type mismatch, but it will not throw an error, but it will be a type mismatch.

"""        
           example:
person2 : Person = {
    "name": "Jane Smith",
    "age": "25,
    "email": "jane.smith@example.com"
}

print(person2)

"""