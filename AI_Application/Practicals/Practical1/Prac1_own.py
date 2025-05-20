import json 
from flask import Flask, request, jsonify

app = Flask(__name__)

employees = [
    {'id': 1, 'name': 'Ashley'},
    {'id': 2, 'name': 'Kate'},
    {'id': 3, 'name': 'Joe'},
    {'id': 4, 'name': 'John'},
    {'id': 5, 'name': 'Doe'},
    {'id': 6, 'name': 'Smith'},
    {'id': 7, 'name': 'Jane'},
    {'id': 8, 'name': 'Emily'},
    {'id': 9, 'name': 'Michael'},
    {'id': 10, 'name': 'Sarah'}
]
nextEmployeeId = 11
@app.route("/employees", methods=["GET"]) # defines the route for the GET request
def get_employees(): # function to handle the GET request and show the employees list in a JSON format. 
    return jsonify(employees)

def get_employee(id):
    return next((i for i in employees if i['id'] == id), None) #searches for the employee list with the given id and returns it

# function is to check if the employee is valid or not
def employee_is_valid(employee):
    for key in employee.keys():
        if key != 'name':
            return False
    return True

# routes the application to the employee list and puts a post function when the user adds a new employee into the list. 
@app.route("/employees", methods=['POST'])
def create_employee():
    global nextEmployeeId
    employee = json.load(request.data)
    if not employee_is_valid(employee): # function is created previously
        return jsonify({'error': 'Invalid employee properties.'}), 400
    employee['id'] = nextEmployeeId
    nextEmployeeId += 1
    employees.append(employee)
    return jsonify(employees), 201

@app.route("/employees/<int:id>", methods=['PUT'])
def update_employee(id: int):
    employee = get_employee(id)
    if employee is None:
        return jsonify({'error': 'Employee does not exist.'}), 404
    
    updated_employee = json.loads(request.data)
    if not employee_is_valid(updated_employee):
        return jsonify({'error': 'Invalid employee properties.'}), 400
    employee.update(updated_employee)
    return jsonify(employee)

@app.route('/employees/<int:id>', methods=['DELETE'])
def delete_employee(id: int):
    global employees
    employee = get_employee(id)
    if employee is None:
        return jsonify({ 'error': 'Employee does not exist.' }), 404

    employees = [e for e in employees if e['id'] != id]
    return jsonify(employee), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)


