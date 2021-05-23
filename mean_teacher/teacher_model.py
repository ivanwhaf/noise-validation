"""
Teacher Model: EMA Model
"""


class TeacherModel(objects):
    """
    Mean Teacher Model class
    """

    def __init__(self, student_model, beta):
        self.model = student_model
        self.beta = beta
        self.teacher = {}  # mean teacher
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.teacher[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                average = self.beta * self.teacher[name] + (1.0 - self.beta) * param.data
                self.teacher[name] = average.clone()

    def apply_teacher(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.teacher[name]

    def restore_student(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
