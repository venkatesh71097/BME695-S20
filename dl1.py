import random
random.seed(0)
import string
size = 5

first, second, last, money = [],[],[], []
def str_gen(size):
    return ''.join(random.choice(string.ascii_lowercase) for x in range(5))

def wealth():
    return random.randint(0, 1000)    

q = [] 
c = []
d = []
h = []
e = []
class People(object):
    
    def __init__(self,first_names,middle_names,last_names,word = None):
        self.first_names = first_names
        self.middle_names = middle_names
        self.last_names = last_names
        self.word = word
        self.index = -1
        
#    def fullname(self):
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index+= 1
        if self.index == 10:
            raise StopIteration
        if self.word == "first_name_first" or self.word == None:
            return q[self.index]
        if self.word == "last_name_first":
            return c[self.index]
        if self.word == "last_name_with_comma_first":
            return d[self.index]
    
    def prin(self):
        print(q)
        
    def __call__(self):
        return sorted(self.last_names)
        
    def gen_names(self):
        '''f = str_gen(size)
        self.first_names.append(f)
        m = str_gen(size)
        self.middle_names.append(m)
        l = str_gen(size)
        self.last_names.append(l)'''
        if self.word == "first_name_first":
            for i in range(10):
                q.append(first[i] + ' ' + second[i] + ' ' + last[i])
        if self.word == "last_name_first":
            for i in range(10):
                c.append(last[i] + ' ' + first[i] + ' ' + second[i])
        if self.word == "last_name_with_comma_first":
            for i in range(10):
                d.append(last[i] + ', ' + first[i] + ' ' + second[i])
      

class PeopleWithMoney(People):
    def __init__(self, wealth):
        super().__init__(first,second,last)
        self.wealth = wealth

    def setwealth(self):
        for i in range(10):
            h.append(money[i])
        
    def __iter__(self):
        People.__iter__(self)
        return self
    
    def __next__(self):
        if self.index == 10:
            raise StopIteration
        return (People.__next__(self) + ' ' + str(h[self.index]))

    def prin(self):
        print(h)
    
    def __call__(self):
        dct = dict((a, b) for a, b in zip(q, h))
        m = sorted(dct.items(), key=lambda x: x[1])
        for i in range(10):
            print(str(m[i][0]) + ' ' + str(m[i][1]))
    
          
first=[str_gen(5) for j in range(10)]
second=[str_gen(5) for j in range(10)]
last=[str_gen(5) for j in range(10)]
p = People(first,second,last,"first_name_first")
a = People(first,second,last,"last_name_first")
b = People(first,second,last,"last_name_with_comma_first")

money = [wealth() for j in range(10)]
r = PeopleWithMoney(money)

for i in range(10):
    p.gen_names()
    a.gen_names()
    b.gen_names()
    r.setwealth()

#first middle last 
for i in range(10):
    print(next(p))

#newline
print()

#last first middle
for i in range(10):
    print(next(a))

#newline
print()

#last,first middle
for i in range(10):
    print(next(b))

#newline
print()

#sorted
a=p()
for i in range(10):
    print(a[i])

#newline
print()

#class & subclass
for k in range(0,10):
    print(next(r))
    
#newline
print()

#sorted class & subclass
r()
