import random

print(random.randint(1,5))

a = 10
def add(x, y):
    return x + y


print(a)
print(add(1, 2))


class Father():
    # int num = 10
    # 双下划綫开始，结束的函数和属性，都是系统已经定义好的成为特殊属性和特殊方法
    # 特殊方法：在特定的场景会自动调用
    # __init__用来初始化对象的属性，因此在创建对象时会自动调用
    def __init__(self, name):
        # self代表当前对象
        print('__init__', id(self))
        self.name = name

    def show(self):
        print('name:',self.name)
        # pass

    def __del__(self):
        # 此方法会在对象销毁时自动执行
        print('__del__')

class Son(Father):
    def __init__(self, name,age):
        super().__init__(name)
        self.age = age

    def show(self):
        print(f'name:{self.name}, age:{self.age}')

if __name__ == "__main__":
    f = Father('张三')
    print(f, id(f))
    f.show()
    del f
    s = Son('AAA',18)
    s.show()
print('-'*100)
print(__name__)