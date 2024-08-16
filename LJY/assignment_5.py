class LogicGate:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

    def compute(self,  *inputs):
        if len(inputs) == self.n_inputs:
            return self.get_output(inputs)
        else:
            raise IndexError("Input index out of range")

    def get_output(self, inputs):
        raise NotImplementedError()


class AndGate(LogicGate):
    def __init__(self):
        super().__init__(2)

    def get_output(self, inputs):
        return all(inputs)


class OrGate(LogicGate):
    def __init__(self):
        super().__init__(2)

    def get_output(self, inputs):
        return any(inputs)


class NotGate(LogicGate):
    def __init__(self):
        super().__init__(1)

    def get_output(self, inputs):
        return not inputs[0]

def main():
    print("选择门：AND, OR, NOT")
    gate_type = input("选择门类型：").strip().upper()

    if gate_type == "AND":
        gate = AndGate()
        input1 = input("输入第一个值（True/False）：").strip().lower() == 'true'
        input2 = input("输入第二个值（True/False）：").strip().lower() == 'true'
        result = gate.compute(input1, input2)
    elif gate_type == "OR":
        gate = OrGate()
        input1 = input("输入第一个值（True/False）：").strip().lower() == 'true'
        input2 = input("输入第二个值（True/False）：").strip().lower() == 'true'
        result = gate.compute(input1, input2)
    elif gate_type == "NOT":
        gate = NotGate()
        input1 = input("输入计算值（True/False）：").strip().lower() == 'true'
        result = gate.compute(input1)

    else:
        print("输入无效")
        return

    print(f"{gate_type} Gate 结果: {result}")
    input("按任意键退出...")

if __name__ == "__main__":
    main()

