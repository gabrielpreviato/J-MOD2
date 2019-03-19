class Classes(object):
    def __init__ (self, class_enum):
        self.class_enum = class_enum

        self.color = (0, 0, 255)

    @staticmethod
    def generate_class(class_enum):
        if class_enum == 0:
            return Robot()
        elif class_enum == 1:
            return  Ball()
        elif class_enum == 2:
            return Goal()

    @staticmethod
    def str_to_class_enum(str):
        if str == 'robot':
            return 0
        elif str == 'ball':
            return 1
        elif str == 'goal':
            return 2


class Robot(Classes):
    def __init__(self):
        super(Robot, self).__init__(0)

        self.color = (255, 0, 0)


class Ball(Classes):
    def __init__(self):
        super(Ball, self).__init__(1)

        self.color = (255, 0, 255)


class Goal(Classes):
    def __init__(self):
        super(Goal, self).__init__(2)

        self.color = (0, 255, 255)
