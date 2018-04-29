#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    DeepCCS: CCS prediction using deep neural network

    Copyright (C) 2018 Pier-Luc

    https://github.com/plpla/DeepCCS

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def split_smiles(smile):
    splitted_smile = []
    for i, letter in enumerate(smile):
        if letter.isupper():
            if smile[i+1].islower():
                splitted_smile.append(smile[i:i+2])
            else:
                splitted_smile.append(letter)
        elif letter.islower() and smile[i - 1].isupper():
            pass
        else:
            splitted_smile.append(letter)
    return splitted_smile
