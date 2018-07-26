import pygame
from pygame.locals import *
import random
import numpy as np
import sys
import time

X = 30
Y = 30

#시작위치 좌표
start_x = 20 
start_y = 17

curr_x = start_x
curr_y = start_y

level_weight = 0
pygame.init()
screen = pygame.display.set_mode([640, 1024])
screen.fill([100, 100, 100])

class gen_v:
    def __init__(self, n , v):
        self.n = n
        self.i = 0
        self.v = v
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            self.i+= 1
            return self.v
        else:
            raise StopIteration

map_list = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1],
  [1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1],
  [1,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1],
  [1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
  [1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
  [1,1,0,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1],
  [1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
  [1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
  [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1],
  [1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
  [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1],
  [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1],
  [1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,1],
  [1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1],
  [1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,1,1,1,1,1],
  [1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

q_table = np.array(list(gen_v(5*30*30,0))).reshape(5, 30, 30)

q_table[0, :, :] = map_list

q_table *= -1 #q 테이블(wall 인 경우 가중치 -1)
print(q_table[0, :, :])

# q_table = np.load("abc.qtb.npy") # 새롭게 시작할 때마다 q_table을 load....


f_map = q_table[0, :, :]

q_table[0, 28, 28] = 1  #(결승점(좌표 : 28, 28) 가중치 1)

# print(q_table)
# np.save('abc.qtb', q_table)
# print(np.load('abc.qtb.npy'))

map_ = np.array(map_list).reshape(X, Y)

wall = pygame.image.load("wall.png")
road = pygame.image.load("road.png")
player = pygame.image.load('player.png')

for i in range(0, map_.shape[0]): #image tile setting
    for j in range(0, map_.shape[1]):
        if map_[i, j] == 1:
            screen.blit(wall, [j*20, i*20])
        elif map_[i, j] == 0:
            screen.blit(road, [j*20, i*20])



def init_f_map(): #f_map matrix init
    f_map  = q_table[0, :, :] #이동 플래그 행렬(q_learn이 재시작 될 때마다 초기화 해야함.)

class select_where:  #selector where
    def __init__(self):
        self.dict = {1:'right', 2:'down', 3:'left', 4:'top'}

    def selector(self, curr_x, curr_y, w):
        self.w = w
        self.select = getattr(self, "select_" + self.dict[self.w])
        return self.select()

    def select_right(self):
        print('right')
        global curr_x, curr_y
        q_table[self.w, curr_x, curr_y] += direction_max(curr_x, curr_y+1)
        curr_y +=1
        f_map[curr_x, curr_y] = 1
        
    def select_down(self):
        print('down')
        global curr_x, curr_y
        q_table[self.w, curr_x, curr_y] += direction_max(curr_x-1, curr_y)
        curr_x -=1
        f_map[curr_x, curr_y] = 1

    def select_left(self):
        print('left')
        global curr_x, curr_y
        q_table[self.w, curr_x, curr_y] += direction_max(curr_x, curr_y-1)
        curr_y -=1
        f_map[curr_x, curr_y] = 1

    def select_top(self):
        print('top')
        global curr_x, curr_y
        q_table[self.w, curr_x, curr_y] += direction_max(curr_x+1, curr_y)
        curr_x +=1
        f_map[curr_x, curr_y] = 1

selector = select_where() #방향, 이동 처리 객체 


#when if f_map ==1  ?
def direction_max(x, y, from_= 0):   #return direction index 
    curr_qt = list(q_table[1:, x, y])
    if from_!=0:
        curr_qt[from_-1] = -999 #왔던 방향은 생각하지 않기
    max_v = np.max(curr_qt)
    if curr_qt.count(max_v) == 1: #action을 할 방향(index+1) 찾기(max(qtable))
        idx = curr_qt.index(max_v)
        print(idx)
    else:
        if max_v ==-999:   #사방이 모두 가중치가 -999인 경우 그만두기 
            np.save('abc.qtb', q_table)  #q_table 저장
            sys.exit()
        index_max_vals = [i for i, val in enumerate(curr_qt) if val == max_v] #max 값이 2개 이상일 때
        idx = random.sample(index_max_vals, 1)[0]
        print(idx)
    return idx+1 #1을 더해서 실제 q_table 상의 index를 맞춤.



def q_learning():
    global curr_x, curr_y
    curr_x = start_x
    curr_y = start_y
    init_f_map()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if q_table[0, curr_x, curr_y] == -1:
            screen.blit(wall, [curr_y, curr_x])
            print('!', curr_y, curr_x)
        elif q_table[0, curr_x, curr_y] == 0:
            screen.blit(road, [curr_y, curr_x])
            print('!', curr_x, curr_y)
        selector.selector(curr_x, curr_y, direction_max(curr_x, curr_y))
        screen.blit(player, [curr_y*20, curr_x*20] )
        print(curr_y, curr_x)
        # print(q_table[0, :, :])
        pygame.time.delay(2000)

        pygame.display.flip()
        pygame.display.update()

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()
#     pygame.display.flip()


if __name__ == "__main__":
    q_learning()
                

        
        
        

    #현재 위치의 q_table 상의 방향 가중치 검사 이동할 위치 결정

    #결정된 이동위치로 이동하기 전 위치 상의 q_table 갱신
    #이동이 제대로 됐는지 확인
    #이동이 제대로 안된 경우 처음 위치부터 다시 시작
    #이동이 제대로 된 경우 현재 위치에서 알고리즘 반복
    pass