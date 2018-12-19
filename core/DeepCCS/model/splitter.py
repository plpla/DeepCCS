#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    DeepCCS: CCS prediction from SMILES using deep neural network

    Copyright (C) 2018 Pier-Luc Plante

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


class SMILESsplitter:
    def split(self, smiles):
        """
        Split a single SMILES using chemical symbols and characters.
        Two letters chemical symbol that end with a 'c' might not be handled properly.
        Nitrogen, Sulfur and Oxygen can miss-handled if they are at the begining of an aromatic structure (ex: Coccc)
        As and Se will be splitted in two caracters if they are found in an aromatic structure.
        Only Co is seen in the current dataset and it is handled properly. TODO: better splitting.
        :param smiles: The SMILES to split
        :return: A list of chemical symbol/character ordered as
        """
        splitted_smiles = []
        for j, k in enumerate(smiles):
            if j == 0:
                if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                    splitted_smiles.append(k + smiles[j + 1])
                else:
                    splitted_smiles.append(k)
            elif j != 0 and j < len(smiles) - 1:
                if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                    splitted_smiles.append(k + smiles[j + 1])
                elif k.islower() and smiles[j - 1].isupper() and k != "c":
                    pass
                else:
                    splitted_smiles.append(k)

            elif j == len(smiles) - 1:
                if k.islower() and smiles[j - 1].isupper() and k != "c":
                    pass
                else:
                    splitted_smiles.append(k)
        return splitted_smiles
