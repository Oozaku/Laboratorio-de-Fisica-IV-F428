#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:56:39 2019

@author: yuki
"""
import numpy as np

class linearRegression:
    def __init__(self):
        self.n = 0
        self.x = []
        self.y = []
        self.w = []
        self.u_y = []
        self.u_x = []
        self.a = 0
        self.b = 0
        self.u_a = 0
        self.u_b = 0
    def calcular(self,x,y,u_x,u_y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.u_y = np.array(u_y)
        self.u_x = np.array(u_x)
        self.n = len(self.y)
        self.w = 1/self.u_y**2
        wx2 = np.sum(np.dot(self.w.T,self.x**2))
        wx = np.dot(self.w.T,self.x)
        wy = np.dot(self.w.T,self.y)
        delta = np.sum(self.w)*wx2 - wx**2
        wxy = 0
        for i in range(self.n):
            wxy += self.w[i]*self.x[i]*self.y[i]
        self.a = (np.sum(self.w)*wxy-wx*wy)/delta
        self.b = (wy*wx2-wxy*wx)/delta
        self.u_a = (np.sum(self.w)/delta)**0.5
        self.u_b = (wx2 / delta)**0.5
        return [self.a,self.u_a,self.b,self.u_b] 
    # Funcoes get
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getU_y(self):
        return self.u_y
    def getU_x(self):
        return self.u_x
    def getN(self):
        return self.n
    def getA(self):
        return self.a
    def getB(self):
        return self.b
    def getU_b(self):
        return self.u_b
    def getU_a(self):
        return self.u_a
    
    
        