from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str ="Aakarsh"
    age:Optional[int]=None
    email: EmailStr
    cgpa:float =Field(gt=0,le=10)

new_student={
    'name': 'Aakarsh',
    'age':19,
    'email':"aaacurse@gmail.com",  
    'cgpa': 9.5

}

student=Student(**new_student)
print(student)

