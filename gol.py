#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import curses
from curses import wrapper

cell_value = lambda world, height, width, y, x: world[y % height, x % width]

row2str = lambda row: ''.join(['o' if c != 0 else '-' for c in row])

def print_world(stdscr, gen, world):
    '''
    盤面をターミナルに出力する
    '''
    
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    height, width = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width -1)
    for y in range(height):
        row = world[y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

def calc_next_cell_state(world, next_world, height, width, y, x):
    cell = cell_value(world, height, width, y, x)
    next_cell = cell
    num = 0
    num += cell_value(world, height, width, y-1, x-1)
    num += cell_value(world, height, width, y-1, x)
    num += cell_value(world, height, width, y-1, x+1)
    num += cell_value(world, height, width, y, x-1)
    num += cell_value(world, height, width, y, x+1)
    num += cell_value(world, height, width, y+1, x-1)
    num += cell_value(world, height, width, y+1, x)
    num += cell_value(world, height, width, y+1, x+1)
    if cell == 0 and num == 3:
        next_cell = 1
    elif cell == 1 and num in (2, 3):
        next_cell = 1
    else:
        next_cell = 0
    next_world[y, x] = next_cell

def calc_next_world(world, next_world):
    '''
    現行世代の盤面の状況をもとに次世代の盤面を計算する
    '''
    height, width = world.shape
    for y in range(height):
        for x in range(width):
            calc_next_cell_state(world, next_world, height, width, y, x)

def gol (stdscr, height, width):
    #状態を持つ2次元配列を生成し、0 or 1の乱数で初期化する。
    world = numpy.random.randint(2, size=(height,width),dtype=numpy.int32)

    gen = 0
    while True:
        print_world(stdscr, gen, world)

        next_world = numpy.empty((height, width), dtype = numpy.int32)
        calc_next_world(world, next_world)
        world = next_world.copy()

        gen += 1

def main(stdscr):
    gol(stdscr, 100,100)

if __name__ == '__main__':
    curses.wrapper(main)
        







