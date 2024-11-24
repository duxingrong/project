#sprite 
import pygame 
import random
import os

import pygame.tests
#变量初始化(大写的字母表示设置了就不会改变)
FPS = 60
WIDTH = 500
HEIGHT = 600
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)
YELLOW = (255,255,0)

#游戏的初始化 and 创建视窗
pygame.init()
pygame.mixer.init()#音校初始化
screen = pygame.display.set_mode((WIDTH,HEIGHT))#窗口的宽度和高度
pygame.display.set_caption("出击")#标题
clock = pygame.time.Clock()


#放入图片
background_img = pygame.image.load(os.path.join("img","background.png")).convert()
player_img = pygame.image.load(os.path.join("img","player.png")).convert()
player_mini_img = pygame.transform.scale(player_img , (25,19))
player_mini_img.set_colorkey(BLACK)
pygame.display.set_icon(player_mini_img)
#rock_img = pygame.image.load(os.path.join("img","rock.png")).convert()
bullet_img = pygame.image.load(os.path.join("img","bullet.png")).convert()
rock_imgs = []
for i in range(7):
    rock_imgs.append(pygame.image.load(os.path.join("img",f"rock{i}.png")).convert())


#加入石头爆炸动画
expl_anim = {}
expl_anim["lg"] = []
expl_anim['sm'] = []
expl_anim['player']= []
for i in range(9):
    expl_img = pygame.image.load(os.path.join("img",f"expl{i}.png")).convert()
    expl_img.set_colorkey(BLACK)
    expl_anim['lg'].append(pygame.transform.scale(expl_img,(75,75)))
    expl_anim['sm'].append(pygame.transform.scale(expl_img,(30,30)))
    player_expl_img = pygame.image.load(os.path.join("img",f"player_expl{i}.png")).convert()
    player_expl_img.set_colorkey(BLACK)
    expl_anim['player'].append(player_expl_img)

power_imgs = {}
power_imgs['shield']= pygame.image.load(os.path.join("img","shield.png")).convert()
power_imgs['gun']= pygame.image.load(os.path.join("img","gun.png")).convert()


#加入音乐
shoot_sound= pygame.mixer.Sound(os.path.join("sound","shoot.wav"))
die_sound =  pygame.mixer.Sound(os.path.join("sound","rumble.ogg"))
gun_sound= pygame.mixer.Sound(os.path.join("sound","pow0.wav"))
shield_sound= pygame.mixer.Sound(os.path.join("sound","pow1.wav"))
expl_sounds = [
    pygame.mixer.Sound(os.path.join("sound","expl0.wav")), 
    pygame.mixer.Sound(os.path.join("sound","expl1.wav"))
]

pygame.mixer.music.load(os.path.join("sound","background.ogg"))
pygame.mixer.music.set_volume(0.4)




#引入字体
font_name = os.path.join("STSong.ttf")
def draw_text(surf,text,size,x,y):
    font = pygame.font.Font(font_name,size)
    text_surface = font.render(text,True,WHITE)
    text_rect = text_surface.get_rect()
    text_rect.centerx = x
    text_rect.top = y
    surf.blit(text_surface,text_rect)

#飞机
class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(player_img,(50,38))
        self.image.set_colorkey(BLACK) #把黑色变透明
 
        self.rect = self.image.get_rect()#进行框选，方便设置
        self.radius = 20
        #pygame.draw.circle(self.image,RED,self.rect.center,self.radius)
        #原点在窗口的左上角,X轴向右，y轴向下
        self.rect.centerx = WIDTH/2
        self.rect.bottom  = HEIGHT-10
        self.speed = 8
        self.health =100
        self.lives = 3
        self.hidden = False
        self.hide_time = 0
        self.gun = 1
        self.gun_time = 0


    def update(self):
        now = pygame.time.get_ticks()
        if self.gun > 1 and now - self.gun_time > 5000:
            self.gun -=1
            self.gun_time = now
        if self.hidden and now - self.hide_time>1000:
            self.hidden = False
            #原点在窗口的左上角,X轴向右，y轴向下 
            self.rect.centerx = WIDTH/2
            self.rect.bottom  = HEIGHT-10
        key_pressed = pygame.key.get_pressed()
        if key_pressed[pygame.K_RIGHT]:
            self.rect.x+=self.speed
        if key_pressed[pygame.K_LEFT]:
            self.rect.x-=self.speed

        if self.rect.right>WIDTH:
            self.rect.right = WIDTH
        if self.rect.left<0:
            self.rect.left = 0
        
    def shoot(self):
        if not self.hidden :
            if self.gun ==1:
                bullet = Bullet(self.rect.centerx,self.rect.top)
                all_sprites.add(bullet)
                bullets.add(bullet)
                shoot_sound.play()
            elif self.gun >= 2:
                bullet1 = Bullet(self.rect.left,self.rect.centery)
                bullet2 = Bullet(self.rect.right,self.rect.centery)
                all_sprites.add(bullet1)
                all_sprites.add(bullet2)
                bullets.add(bullet1)
                bullets.add(bullet2)
                shoot_sound.play()                

    def hide(self):
        self.hidden = True
        self.hide_time = pygame.time.get_ticks()
        self.rect.center = (WIDTH/2,HEIGHT+500)

    def gunup(self):
        self.gun +=1
        self.gun_time = pygame.time.get_ticks()


class Rock(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image_ori = random.choice(rock_imgs)
        self.image_ori.set_colorkey(BLACK)
        self.image = self.image_ori.copy() 

        self.rect = self.image.get_rect()#进行框选，方便设置
        self.radius = int(self.rect.width*0.85/2)
        #pygame.draw.circle(self.image,RED,self.rect.center,self.radius)
        #原点在窗口的左上角,X轴向右，y轴向下
        self.rect.x = random.randrange(0,WIDTH-self.rect.width)
        self.rect.y = random.randrange(-180,-100)
        self.speedy = random.randrange(2,10) 
        self.speedx = random.randrange(-3,3) 
        self.total_degree = 0
        self.rot_degree = random.randrange(-3,3)

    def rotate(self):
        self.total_degree+=self.rot_degree
        self.total_degree = self.total_degree%360
        self.image = pygame.transform.rotate(self.image_ori,self.total_degree)
        #重新定位并且中心点不变
        center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = center

    def update(self):
        self.rotate()
        self.rect.y +=self.speedy
        self.rect.x+=self.speedx
        #超出画面就重重
        if self.rect.top>HEIGHT or self.rect.left>WIDTH or self.rect.right<0:
            self.rect.x = random.randrange(0,WIDTH-self.rect.width)
            self.rect.y = random.randrange(-100,-40)
            self.speedy = random.randrange(2,10) 
            self.speedx = random.randrange(-3,3)            
 

class Bullet(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.image.set_colorkey(BLACK)

        self.rect = self.image.get_rect()#进行框选，方便设置
        #原点在窗口的左上角,X轴向右，y轴向下
        self.rect.centerx = x
        self.rect.bottom = y
        self.speedy = -10

        

    def update(self):
        self.rect.y +=self.speedy
        if self.rect.bottom<0:
            self.kill()

#宝物
class Power(pygame.sprite.Sprite):
    def __init__(self,center):
        pygame.sprite.Sprite.__init__(self)
        self.type  = random.choice(['shield','gun'])
        self.image = power_imgs[self.type]
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.speedy = 3
 

    def update(self):
        self.rect.y +=self.speedy
        if self.rect.top>HEIGHT:
            self.kill()

#加入爆炸动画
class Explosion(pygame.sprite.Sprite):
    def __init__(self,center,size):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self.image = expl_anim[self.size][0]
        self.rect = self.image.get_rect()

        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 50

        

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame+=1
            if self.frame == len(expl_anim[self.size]):
                self.kill()
            else:
                self.image = expl_anim[self.size][self.frame]
                center = self.rect.center
                self.rect = self.image.get_rect()
                self.rect.center = center
#创建石头
def new_rock():
    r = Rock()
    all_sprites.add(r)
    rocks.add(r)


#画生命值
def draw_health(surf,hp,x,y):
    if hp<0 :
        hp = 0
    BAR_LENGTH = 100
    BAR_HEIGHT = 10
    fill = (hp/100)*BAR_LENGTH
    outline_rect = pygame.Rect(x,y,BAR_LENGTH,BAR_HEIGHT)
    fill_rect = pygame.Rect(x,y,fill,BAR_HEIGHT)
    pygame.draw.rect(surf,GREEN,fill_rect)
    pygame.draw.rect(surf,WHITE,outline_rect,2)

#小飞船
def draw_lives (surf,lives,img,x,y):
    for i in range(lives):
        img_rect = img.get_rect()
        img_rect.x = x+32*i
        img_rect.y = y
        surf.blit(img,img_rect)



def draw_init():
    screen.blit(background_img,(0,0))
    draw_text(screen, "太空生存战!", 64,WIDTH/2,HEIGHT/4)
    draw_text (screen, "<左 右>  移动飞船  空白键发射子弹", 22, WIDTH/2, HEIGHT/2)
    draw_text (screen, "按任意键开始游戏", 18, WIDTH/2, HEIGHT*3/4)
    pygame.display.update()
    waiting = True
    while waiting:
        clock.tick(FPS)#while一秒中最多只会执行10次
        #取得玩家的输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            elif event.type == pygame.KEYUP:
                waiting = False
                return False




all_sprites = pygame.sprite.Group()
rocks  = pygame.sprite.Group()
bullets = pygame.sprite.Group()
player  = Player()
powers = pygame.sprite.Group()

all_sprites.add(player)
for i in range(8):
    new_rock()
        
score = 0#分数
pygame.mixer.music.play(-1)#-1代表无限重复播放




#游戏回圈
show_init = True
running = True
while running:
    if show_init:
        close = draw_init()
        if close:
            break
        show_init = False
    clock.tick(FPS)#while一秒中最多只会执行10次
    #取得玩家的输入
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running= False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.shoot()

    # 更新游戏
    all_sprites.update()
    #hits返回的字典key是rocks,value是bullets
    hits= pygame.sprite.groupcollide(rocks,bullets,True,True)#查看两个群组是否碰撞了，并且设置碰撞后是否删除
    for hit in hits:
        random.choice(expl_sounds).play()
        score+= hit.radius
        expl = Explosion(hit.rect.center,'lg')
        all_sprites.add(expl)
        if random.random()> 0.92:
            pow = Power(hit.rect.center)
            all_sprites.add(pow)
            powers.add(pow)
        new_rock()

    hits = pygame.sprite.spritecollide(player,rocks,True,pygame.sprite.collide_circle)#碰撞改成圆形的
    #传回来一个石头的列表
    for hit in  hits:
        new_rock()
        player.health-= hit.radius
        expl = Explosion(hit.rect.center,'sm')
        all_sprites.add(expl)
        if player.health <=0:
            death_expl = Explosion(player.rect.center,'player')
            all_sprites.add(death_expl)
            die_sound.play()
            player.lives-=1
            player.health =100 
            player.hide()

    #判断宝物和飞船相撞
    hits= pygame.sprite.spritecollide(player,powers,True,pygame.sprite.collide_circle)
    for hit in hits :
        if hit.type == 'shield':
            player.health +=20
            if player.health>100:
                player.health = 100
            shield_sound.play()
        elif hit.type == 'gun':
            player.gunup()
            gun_sound.play()



    if player.lives == 0 and not(death_expl.alive()):
        show_init = True
        all_sprites = pygame.sprite.Group()
        rocks  = pygame.sprite.Group()
        bullets = pygame.sprite.Group()
        player  = Player()
        powers = pygame.sprite.Group()
        all_sprites.add(player)
        for i in range(8):
            new_rock()
        score = 0#分数


    # 画面显示
    screen.fill(BLACK) #调色盘RGB
    screen.blit(background_img,(0,0))
    all_sprites.draw(screen)
    draw_text(screen,str(score),18,WIDTH/2,10)
    draw_health(screen,player.health,5,15)
    draw_lives(screen,player.lives,player_mini_img,WIDTH-100 , 15)
    pygame.display.update()#更新画面

    
pygame.quit()
