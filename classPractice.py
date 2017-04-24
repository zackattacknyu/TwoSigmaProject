print("Hello class practice")



class animal:

    def __init__(self,speed):
        self.speed = speed

    def updateSpeed(self):
        self.speed = self.speed+2




dog1 = animal(4)
dog1.updateSpeed()

print(dog1.speed)