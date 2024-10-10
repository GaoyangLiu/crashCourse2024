class LogicGate:
    def __init__(self, label):
        self.label = label
        self.pinA = None
        self.pinB = None
        self.output = None

    def set_pinA(self, pin):
        self.pinA = pin

    def set_pinB(self, pin):
        self.pinB = pin

    def get_output(self):
        self.perform_gate_logic()
        return self.output

    def perform_gate_logic(self):
        raise NotImplementedError("Must be implemented by subclasses")

class AndGate(LogicGate):
    def __init__(self, label):
        super().__init__(label)

    def perform_gate_logic(self):
        if self.pinA is not None and self.pinB is not None:
            self.output = 1 if self.pinA and self.pinB else 0

class OrGate(LogicGate):
    def __init__(self, label):
        super().__init__(label)

    def perform_gate_logic(self):
        if self.pinA is not None and self.pinB is not None:
            self.output = 1 if self.pinA or self.pinB else 0

class NotGate(LogicGate):
    def __init__(self, label):
        super().__init__(label)
        self.pinA = None  # NOT Gate only has one input

    def set_pin(self, pin):
        self.pinA = pin

    def perform_gate_logic(self):
        if self.pinA is not None:
            self.output = 0 if self.pinA else 1

def main():
    # 创建逻辑门实例
    and_gate = AndGate('G1')
    or_gate = OrGate('G2')
    not_gate = NotGate('G3')

    # 获取用户输入
    and_gate.set_pinA(int(input(f"Enter input for AND gate (0 or 1): ")))
    and_gate.set_pinB(int(input(f"Enter input for AND gate (0 or 1): ")))
    or_gate.set_pinA(int(input(f"Enter input for OR gate (0 or 1): ")))
    or_gate.set_pinB(int(input(f"Enter input for OR gate (0 or 1): ")))
    not_gate.set_pin(int(input(f"Enter input for NOT gate (0 or 1): ")))

    # 获取输出
    print(f"The output of AND gate is: {and_gate.get_output()}")
    print(f"The output of OR gate is: {or_gate.get_output()}")
    print(f"The output of NOT gate is: {not_gate.get_output()}")

if __name__ == "__main__":

    main()