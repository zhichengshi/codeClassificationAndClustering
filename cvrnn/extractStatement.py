import sys
import xml.etree.ElementTree as ET

'''
根据函数体源码生成的xml提取statement子树序列
'''


# 带双亲节点的树节点
class treeNode:
    def __init__(self, parent, ele):
        if parent != None:
            self.parent = parent
            self.ele = ele
        else:
            self.parent = parent
            self.ele = ele


# 根据根节点提取AST
def extractSTBaseRoot(root):
    # 添加AST叶子节点
    def transform(root):
        if root.text != None:
            root.append(ET.Element(root.text))
        for child in root:
            transform(child)
        return root

    # 深度优先遍历树
    def traverse(node):
        print(node.tag)
        for childNode in node:
            traverse(childNode)

    # 根据深度优先遍历得到的列表，提取statement子树
    def extractStatement(tree):
        statementList = []
        for node in tree:
            if node.ele.tag in statemnentTag:
                statementList.append(node.ele)
                if node.parent != None:
                    node.parent.remove(node.ele)
        return statementList

    # 深度优先遍历树，树的节点为带双亲节点的结构
    def createTreeDeepFirst(root, list, parent):
        list.append(treeNode(parent, root))
        for node in root:
            createTreeDeepFirst(node, list, root)

    statemnentTag = {"if", "while", "for", "unit", "switch"}
    treeDeepFirstList = []
    # root = transform(root)
    createTreeDeepFirst(root, treeDeepFirstList, None)
    statementList = extractStatement(treeDeepFirstList)
    return statementList
