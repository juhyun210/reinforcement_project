import pygame
from pygame.locals import *
import random
import numpy as np
import sys
import time

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
"""
map_list = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1],
  [1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1],
  [1,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,1,1],
  [1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
  [1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
  [1,1,0,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
  [1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1],
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
  [1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
  """
map_list = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

### q_tabel map  ###
### wall = -1
### road = 0


pygame.init()
screen = pygame.display.set_mode([640, 1024])
screen.fill([100, 100, 100])

I = 30
C = 30

#시작위치 좌표
start_i = 1 
start_c = 1 

curr_i = start_i
curr_c = start_c

# level_weight = 0


#make q_table 
def init_q_table():
    try:
        q_table = np.load('abc.qtb.npy')
    except FileNotFoundError as e:
        q_table = np.array(list(gen_v(5*I*C,0))).reshape(5, I, C)
        q_table[0, :, :] = map_list
        q_table *= -1 #q 테이블(wall 인 경우 가중치 -1)
        q_table[0, 28, 28] = 1  #(결승점(좌표 : 28, 28) 가중치 1)
    else:
        sys.exit()
    finally:
        return q_table
    
# q_table = np.array(list(gen_v(5*30*30,0))).reshape(5, 30, 30)
# q_table[0, :, :] = map_list
# q_table *= -1 #q 테이블(wall 인 경우 가중치 -1)
# q_table[0, 28, 28] = 1  #(결승점(좌표 : 28, 28) 가중치 1)

# print(q_table[0, :, :])


# q_table = np.load("abc.qtb.npy") # 새롭게 시작할 때마다 q_table을 load....




# print(q_table)
# np.save('abc.qtb', q_table)
# print(np.load('abc.qtb.npy'))

# map_ = np.array(map_list).reshape(I, C)

### 맵 초기화
road = pygame.image.load(b"road.png")
wall = pygame.image.load(b"wall.png")
player = pygame.image.load(b'player.png')
stop = pygame.image.load('stop.png')

q_table = init_q_table()

f_map = q_table[0, :, :].copy() # for문을 위한 임시 변수  
# f_map = np.array(map_list)

for i in range(0, f_map.shape[0]): #image tile setting
    for j in range(0, f_map.shape[1]):
        if f_map[i, j] == -1:
            screen.blit(wall, [j*20, i*20])
        elif f_map[i, j] == 0:
            screen.blit(road, [j*20, i*20])



def init_f_map(): #f_map matrix init
    global f_map
    f_map  = q_table[0, :, :] #이동 플래그 행렬(q_learn이 재시작 될 때마다 초기화 해야함.)

class select_where:  #selector where
    def __init__(self):
        self.dict = {1:'right', 2:'down', 3:'left', 4:'top'}

    def selector(self, w):
        self.w = w
        self.select = getattr(self, "select_" + self.dict[self.w])
        return self.select()

    def select_right(self):
        global curr_i, curr_c
        # print(curr_i, curr_c)
        # print('right')
        if q_table[0,curr_i, curr_c+1] == -1:
            screen.blit(stop,[(curr_c+1)*20, curr_i*20])
            q_table[self.w, curr_i, curr_c] += -2
            return True           # 게임 루프 종료 
        if q_table[0,curr_i, curr_c+1] == 1:
            q_table[self.w, curr_i, curr_c] += 1
            return True
        q_table[self.w, curr_i, curr_c] += np.sign(np.max(list(q_table[1:,curr_i, curr_c+1])))
        curr_c +=1
        f_map[curr_i, curr_c] = 1
        
    def select_down(self):
        global curr_i, curr_c
        # print(curr_i, curr_c)
        # print('down')
        if q_table[0,curr_i+1, curr_c] == -1:
            screen.blit(stop,[(curr_c)*20, (curr_i+1)*20])
            q_table[self.w, curr_i, curr_c] += -2
            return True           # 게임 루프 종료 
        if q_table[0,curr_i+1, curr_c] == 1:
            q_table[self.w, curr_i, curr_c] += 1
            return True
        q_table[self.w, curr_i, curr_c] += np.sign(np.max(list(q_table[1:,curr_i+1, curr_c])))
        curr_i +=1
        f_map[curr_i, curr_c] = 1

    def select_left(self):
        global curr_c, curr_i
        # print(curr_i, curr_c)
        # print('left')
        if q_table[0,curr_i, curr_c-1] == -1:
            screen.blit(stop,[(curr_c-1)*20, curr_i*20])
            q_table[self.w, curr_i, curr_c] += -2
            return True           # 게임 루프 종료 
        if q_table[0,curr_i, curr_c-1] == 1:
            q_table[self.w, curr_i, curr_c] += 1
            return True
        q_table[self.w, curr_i, curr_c] += np.sign(np.max(list(q_table[1:,curr_i, curr_c-1])))
        curr_c -=1
        f_map[curr_i, curr_c] = 1

    def select_top(self):
        global curr_i, curr_c
        # print(curr_i, curr_c)
        # print('top')
        if q_table[0,curr_i-1, curr_c] == -1:
            screen.blit(stop,[curr_c*20, (curr_i-1)*20])
            q_table[self.w, curr_i, curr_c] += -2
            return True           # 게임 루프 종료 
        if q_table[0,curr_i-1, curr_c] == 1:
            q_table[self.w, curr_i, curr_c] += 1
            return True
        q_table[self.w, curr_i, curr_c] += np.sign(np.max(list(q_table[1:,curr_i-1, curr_c])))
        curr_i -= 1
        f_map[curr_i, curr_c] = 1

selector = select_where() #방향, 이동 처리 객체 

previous_direction = 0

#when if f_map ==1  ?
def direction_max(x, y, from_= 0):   #return direction index 
    global previous_direction
    curr_qt = list(q_table[1:, x, y])
    if from_!=0:
        curr_qt[(from_+1)%4] = -999 #왔던 방향은 생각하지 않기
    max_v = np.max(curr_qt)
    if curr_qt.count(max_v) == 1: #action을 할 방향(index+1) 찾기(max(qtable))
        idx = curr_qt.index(max_v)
        previous_direction = idx+1
        # print(idx)
    else:
        # if max_v ==-999:   #사방이 모두 가중치가 -999인 경우 그만두기 
        #     np.save('abc.qtb', q_table)  #q_table 저장
        #     sys.exit()
        index_max_vals = [i for i, val in enumerate(curr_qt) if val == max_v] #max 값이 2개 이상일 때
        idx = random.sample(index_max_vals, 1)[0]
        previous_direction = idx+1
        # print(idx)
    return idx+1 #1을 더해서 실제 q_table 상의 index를 맞춤.

def q_learning(train_num):
    init_q_table()
    global curr_i, curr_c
    curr_i = start_i
    curr_c = start_c
    print("%d 번째 훈련"% train_num)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if q_table[0, curr_i, curr_c] == -1:
            screen.blit(wall, [curr_c*20, curr_i*20])
            # print('!1--', curr_c, curr_i)
        elif q_table[0, curr_i, curr_c] == 0:
            screen.blit(road, [curr_c*20, curr_i*20])
            # print('!2--', curr_c, curr_i)

        if selector.selector(direction_max(curr_i, curr_c, previous_direction)):  #q_table update
            print("end")
            break
        # print('pd', previous_direction, end='\n\n')
        screen.blit(player, [curr_c*20, curr_i*20] )      #screen update
        # print(curr_c, curr_i)
        # print(q_table[0, :, :])
        pygame.time.delay(1 )

        pygame.display.flip()
        pygame.display.update()
    np.save('abc.qtb', q_table)

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()
#     pygame.display.flip()


if __name__ == "__main__":
    for i in range(0, 1000):
        q_learning(i)
                

        
        
        

    #현재 위치의 q_table 상의 방향 가중치 검사 이동할 위치 결정

    #결정된 이동위치로 이동하기 전 위치 상의 q_table 갱신
    #이동이 제대로 됐는지 확인
    #이동이 제대로 안된 경우 처음 위치부터 다시 시작
    #이동이 제대로 된 경우 현재 위치에서 알고리즘 반복
    pass