import torch
from torch import nn
from torch.nn import functional as F

def sigmoid_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    temperature = float(temperature)
    if temperature == 1.0:
        targets = torch.sigmoid(teacher_logits.detach().float())
        return F.binary_cross_entropy_with_logits(student_logits.float(), targets)
    targets = torch.sigmoid(teacher_logits.detach().float() / temperature)
    return temperature ** 2 * F.binary_cross_entropy_with_logits(student_logits.float() / temperature, targets)

class SyntheticKnowledgeTrainingModel(nn.Module):

    def __init__(self, student, teacher, weight, student_mean, student_std, teacher_mean, teacher_std, temperature=1.0):
        super().__init__()
        self.student = student
        self.teacher = teacher.requires_grad_(False).eval()
        self.weight = float(weight)
        self.temperature = float(temperature)
        stats = [torch.as_tensor(x, dtype=torch.float32).reshape(1, -1, 1) for x in (student_mean, student_std, teacher_mean, teacher_std)]
        for (name, value) in zip(('student_mean', 'student_std', 'teacher_mean', 'teacher_std'), stats):
            self.register_buffer(name, value)
        self._sk_loss = None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self._sk_loss = None
        return self

    def forward(self, signals):
        logits = self.student(signals)
        self._sk_loss = None
        if self.training:
            with torch.no_grad(), torch.amp.autocast(device_type=signals.device.type, enabled=False):
                teacher_input = (signals.float() * self.student_std + self.student_mean - self.teacher_mean) / self.teacher_std
                teacher_logits = self.teacher(teacher_input)
            self._sk_loss = sigmoid_distillation_loss(logits, teacher_logits, self.temperature)
        return logits

    def auxiliary_loss(self, labels):
        native_fn = getattr(self.student, 'auxiliary_loss', None)
        native_loss = native_fn(labels) if native_fn is not None else None
        if not self.training or self._sk_loss is None:
            return native_loss
        sk = self.weight * self._sk_loss
        return sk if native_loss is None else native_loss + sk

def inference_model(model):
    return model.student if isinstance(model, SyntheticKnowledgeTrainingModel) else model
