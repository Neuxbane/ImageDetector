import pygame
import random
import math
import time
def HSVtoRGB(color,dark,white):
	color = (color-(round(color/360)+(((color/360)>round(color/360))*1)-1)*360)
	R = (((int(color>=0) == int(120>=color)) or (int(color>=300) == int(360>=color)))*255)+(((int(color>=60) == int(120>=color))*((60-color)*(255/60))))+((int(color>240) == int(300>color))*((color-240)*(255/60)))
	G = (((int(color>=0) == int(60>=color))*((color)*(255/60))))+((int(color>60) == int(240>=color))*255)+((int(color>=180) == int(240>=color))*((180-color)*(255/60)))
	B = (((int(color>=120) == int(180>=color))*((color-120)*(255/60))))+(((int(color>180) == int(360>color))*255))+((((int(color>300) == int(360>color))*(300-color)*(255/60))))
	if not R == 255:
		R = (255 if (((100-dark)/100)*255)+R>=255 else R+(((100-dark)/100)*255))
	if not G == 255:
		G = (255 if (((100-dark)/100)*255)+G>=255 else G+(((100-dark)/100)*255))
	if not B == 255:
		B = (255 if (((100-dark)/100)*255)+B>=255 else B+(((100-dark)/100)*255))
	R = R*(white/100)
	G = G*(white/100)
	B = B*(white/100)
	return (R, G, B)

def getimage(path):
	global pygame
	global window
	pygame.init()
	window = pygame.display.set_mode((100, 100))
	try:
		image = pygame.image.load(path)
		image = pygame.transform.scale(image, (100,100))
		window.blit(image,(0,0))
		data = []
		for a in range(100*100):
			y = int((round(a/100) if (a/100)>=round(a/100) else round(a/100)-1))
			x = int(((a-(round(a/100)*100) if (a/100)>=round(a/100) else a-((round(a/100)-1)*100))))
			n = (eval('+'.join([f"{x}" for x in window.get_at((x,y))[:3]]))/765)*100
			data.append(n)
			pygame.draw.rect(window,HSVtoRGB(n,100,100),(x,y,1,1))
		pygame.display.flip()
		pygame.image.save(window,f'Input.png')
	except Exception as e:
		pass
	pygame.quit()
	return data

def Sigmoid(x):
	try:
		return 1 / (1 + math.exp(-x))
	except:
		return x*(0<=x<=1)
TransferDerivative = lambda x: x * (1.0 - x)
ReLU = lambda x: x * (x > 0)
rand = lambda a,b: (b-a)*random.random() + a
neux = lambda x: (-1*(x<0.0))+(x>0.0)+(x==0)
class NeuralNetwork:
	def __init__(self,inp,hidlyr,hid,outp,activation='Sigmoid',n=1,minloss=0.4,iterations=-1):
		self.seed = 1603445364#round(time.time())
		random.seed(self.seed)
		counter = time.time()
		self.iterations = iterations
		self.minloss = minloss
		self.testresult = []
		self.n = n
		self.activation = activation
		self.inp = [0 for i in range(inp)]
		self.h = [[0 for i in range(hid)] for i in range(hidlyr)]
		self.outp = [0 for i in range(outp)]
		self.roundoutp = [0 for i in range(outp)]
		self.hw = [[[0 for i in range(hid)] for i in range(hid)] for i in range(hidlyr-1)]
		self.inpw = [[0 for i in range(hid)] for i in range(inp)]
		self.outpw = [[0 for i in range(hid)] for i in range(outp)]
		self.outpb = [0 for i in range(outp)]
		self.hb = [[0 for i in range(hid)] for i in range(hidlyr-1)]
		self.inpb = [0 for i in range(inp)]
		self.random(1)
		#print('\n','Time Spend to Init Network ',time.time()-counter,'\n\n input		=',self.inp,'\n\n','hidden 	= ',self.h,'\n\n','output 	=',self.outp,'\n\n','\n\n weight input to hidden		=',self.inpw,'\n\n','weight hidden 			= ',self.hw,'\n\n','weight hidden to output 	=',self.outpw,'\n\n')
	def calculate(self,a):
		self.h = [[0 for i in range(len(self.h[0]))] for i in range(len(self.h))]
		self.outp = [0 for i in range(len(self.outp))]
		for b in range(len(self.inp)):
			for c in range(len(self.inpw[0])):
				self.inp[b] = a[b]
				self.h[0][c] = (self.h[0][c]+((self.inp[b]*self.inpw[b][c])+self.inpb[b]))
		for b in range(len(self.h)-1):
			for c in range(len(self.hw[0])):
				for d in range(len(self.hw[0][0])):
					self.h[b+1][c] = (self.h[b+1][c]+((self.h[b][c]*self.hw[b][c][d])+self.hb[b][c]))
		for b in range(len(self.outp)):
			for c in range(len(self.outpw[0])):
				self.outp[b] = self.outp[b]+((self.h[len(self.h)-1][c]*self.outpw[b][c])+self.outpb[b])
		exec('self.outp = ['+self.activation+'(x) for x in self.outp]')
		try:
			self.roundoutp = [round(b) for b in self.outp]
		except:
			pass
		self.testresult.append(self.roundoutp)
	def random(self,a):
		for b in range(len(self.h)-1):
			for c in range(len(self.h[0])):
				self.hb[b][c] = (rand(-1*a,a))
				for d in range(len(self.h[0])):
					self.hw[b][c][d] = (rand(-1*a,a))
		for b in range(len(self.inp)):
			self.inpb[b] = (rand(-1*a,a))
			for c in range(len(self.h[0])):
				self.inpw[b][c] = (rand(-1*a,a))
		for b in range(len(self.outp)):
			self.outpb[b] = (rand(-1*a,a))
			for c in range(len(self.h[0])):
				self.outpw[b][c] = (rand(-1*a,a))
	def getloss(self,pola):
		loss = 0
		for a in pola:
			self.calculate(a[0])
			for b in range(len(self.outp)):
				loss += abs((a[1][b]-self.outp[b])*(a[1][b]-self.outp[b]))
		return loss
	def train(self,pola,show=True):
		speed = self.n
		counter = time.time()
		OLoss = 1
		literation = 0
		self.oldloss = 9999999
		self.oldData = [self.inpw,self.hw,self.outpw,self.inpb,self.hb,self.outpb]
		for a in pola:
			self.calculate(a[0])
		while not (((self.getloss(pola)<self.minloss)) or (self.iterations <= literation and self.iterations != -1)):
			try:
				if show:
					print('literation',literation,'loss',self.getloss(pola))
				if 'e+' in str(self.getloss(pola)) or 'nan' in str(self.getloss(pola)):
					self.n -= self.n/50
					self.random(1)
					if show:
						print(f'------------------------------------------:set Learning Rate to {self.n}')
				if self.getloss(pola) < self.oldloss:
					self.oldloss = self.getloss(pola)
					self.oldData = [eval(str(self.inpw)+','+str(self.hw)+','+str(self.outpw)+','+str(self.inpb)+','+str(self.hb)+','+str(self.outpb))]
				for b in range(len(self.h)-1):
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.hb[b][c] += 1e-10
						speed = (OLoss-self.getloss(pola))*self.n
						self.hb[b][c] -= 1e-10
						self.hb[b][c] += speed
						for d in range(len(self.h[0])):
							OLoss = self.getloss(pola)
							self.hw[b][c][d] += 1e-10
							speed = (OLoss-self.getloss(pola))*self.n
							self.hw[b][c][d] -= 1e-10
							self.hw[b][c][d] += speed
				for b in range(len(self.inp)):
					OLoss = self.getloss(pola)
					self.inpb[b] += 1e-10
					speed = (OLoss-self.getloss(pola))*self.n
					self.inpb[b] -= 1e-10
					self.inpb[b] += speed
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.inpw[b][c] += 1e-10
						speed = (OLoss-self.getloss(pola))*self.n
						self.inpw[b][c] -= 1e-10
						self.inpw[b][c] += speed
				for b in range(len(self.outp)):
					OLoss = self.getloss(pola)
					self.outpb[b] += 1e-10
					speed = (OLoss-self.getloss(pola))*self.n
					self.outpb[b] -= 1e-10
					self.outpb[b] += speed
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.outpw[b][c] += 1e-10
						speed = (OLoss-self.getloss(pola))*self.n
						self.outpw[b][c] -= 1e-10
						self.outpw[b][c] += speed
				OLoss = self.getloss(pola)
				literation += 1
			except KeyboardInterrupt as e:
				if show:
					print('Litaration End Because KeyboardInterrupt')
					print(e)
				break
		if self.getloss(pola) > self.oldloss:
			self.data_set(self.oldData[0][0],self.oldData[0][1],self.oldData[0][2],self.oldData[0][3],self.oldData[0][4],self.oldData[0][5])
		if show:
			print('DATA-----------------')
			print(str(self.inpw)+','+str(self.hw)+','+str(self.outpw)+','+str(self.inpb)+','+str(self.hb)+','+str(self.outpb))
			print('END------------------')
			print('literation',literation,'loss',self.getloss(pola))
			print('Random seed : ',self.seed)
		open('this.data','w').write(f"{self.inpw},{self.hw},{self.outpw},{self.inpb},{self.hb},{self.outpb}")
	def test(self,a,r=0):
		counter = time.time()
		if r:
			self.Reverse_calculate(a)
			return [self.inp,self.roundoutp,time.time()-counter]
		else:
			self.calculate(a)
			return [self.outp,self.roundoutp,time.time()-counter]
	def data_set(self,winput,whidden,woutput,binput,bhidden,boutput):
		self.inpw = winput
		self.outpw = woutput
		self.hw = whidden
		self.inpb = binput
		self.hb = bhidden
		self.outpb = boutput

def result(a):
	global classes
	return [1 if x==a else 0 for x in classes]


#input = ... , hiddenlayer = ... , node per hiddenlayer = ... , output = ...
a = NeuralNetwork(10000,1,3,2,activation='Sigmoid',n=1,minloss=-1,iterations=-1)
r = eval(open('this.data','r').read())
a.data_set(r[0],r[1],r[2],r[3],r[4],r[5])
classes = ['bird','cat']
a.train([
	[getimage('DataSetImage/bird1.jpg'),result('bird')],
	[getimage('DataSetImage/bird2.jpg'),result('bird')],
	[getimage('DataSetImage/bird3.jpg'),result('bird')],
	[getimage('DataSetImage/cat1.jpg'),result('cat')],
	[getimage('DataSetImage/cat2.jpg'),result('cat')],
	[getimage('DataSetImage/cat3.jpg'),result('cat')]
	])
while True:
	try:
		c = getimage(input('|Input<'))
		b = a.test(c)
		print(f"|Output>{' , '.join([f'{classes[x]} {round((b[0][x])*100)}%' if round(b[1][x])!= 0 else f'' for x in range(len(b[1]))])}")
	except Exception as e:
		print(e)