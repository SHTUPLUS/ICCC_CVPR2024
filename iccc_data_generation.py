import queue
import json
import spacy
from tqdm import tqdm
from collections import *
import sys


class TreeNode:
    def __init__(self, idx, curr, ent_idx=None, rel_idx=None):
        self.init_token = curr
        self.id = idx
        self.ent_idx = ent_idx
        self.rel_idx = rel_idx
        self.text = curr.text
        self.dep = curr.dep_
        self.pos = curr.pos_

        self.parent_id = None
        self.parent_token = None
        self.children_id = []
        self.children_token = []

    def add_child(self, child):
        self.children_id.append(child)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.id}, {self.text} {self.dep} {self.pos} {self.parent_id} "

def bfs(start_node, match_cond):
    q = queue.Queue()
    q.put(start_node)
    while q.qsize() > 0:
        curr = q.get()
        if match_cond(curr):
            return curr
        for each in curr.children_token:
            q.put(each)

def get_node_id(node):
    return node.id

def is_obj(node):
    if ('obj' in node.dep or ('acomp' == node.dep and node.pos == 'ADJ') or 'xcomp' in node.dep or 'advmod' in node.dep) and node.dep != 'ROOT':
        return True
    else:
        return False

def is_obj_only(node):
    if ('obj' in node.dep and node.dep != 'ROOT'):
        return True
    else:
        return False

def is_sub(node):
    if 'sub' in node.dep:
        return True
    else:
        return False

def is_attr(node):
    if ('attr' in node.dep):
        return True
    else:
        return False

def get_all_dep_pred(start_node, degree=-1):
    if start_node is None:
        return None
    
    all_children = []
    q = queue.Queue()
    q.put(start_node)
    curr_deg = 0
    while q.qsize() > 0:
        curr = q.get()
        curr_deg += 1
        for each_cld in curr.children_token:
            if each_cld.pos == 'ADP' or ('obj' not in each_cld.dep and 'sub' not in each_cld.dep) and each_cld.dep != 'prep':
                if curr_deg < degree and degree > 0:
                    q.put(each_cld)
                if each_cld.dep != 'punct':
                    all_children.append(each_cld)
    all_children.append(start_node)
    all_children = sorted(all_children, key=get_node_id)
    return all_children

def get_all_dep_ent(start_node, degree=-1):
    if start_node is None:
        return None
    
    all_children = []
    q = queue.Queue()
    q.put(start_node)
    curr_deg = 0
    while q.qsize() > 0:
        curr = q.get()
        curr_deg += 1
        for each_cld in curr.children_token:
            # print(curr.children_token)
            if each_cld.dep != 'prep' and 'obj' not in each_cld.dep and 'sub' not in each_cld.dep :
                if curr_deg >= degree and degree > 0:
                    q.put(each_cld)
                if each_cld.dep != 'punct':
                    all_children.append(each_cld)
    all_children.append(start_node)
    all_children = sorted(all_children, key=get_node_id)
    return all_children

def __flatten_conjunction(node):
    yield node
    for c in node.children:
        if c.dep_ == 'conj':
            yield c

def get_entity(doc):
    entities = list()
    entity_chunks = list()
    for entity in doc.noun_chunks:
        # Ignore pronouns such as "it".
        if entity.root.lemma_ == '-PRON-':
            continue

        ent = dict(
            span=entity.text,
            lemma_span=entity.lemma_,
            head=entity.root.text,
            lemma_head=entity.root.lemma_,
            span_bounds=(entity.start, entity.end),
            modifiers=[]
        )

        def dfs(node):
            for x in node.children:
                if x.dep_ == 'det':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'nummod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                elif x.dep_ == 'amod':
                    for y in __flatten_conjunction(x):
                        ent['modifiers'].append({'dep': x.dep_, 'span': y.text, 'lemma_span': y.lemma_})
                elif x.dep_ == 'compound':
                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']
                    dfs(x)
        dfs(entity.root)

        entities.append(ent)
        entity_chunks.append(entity)
    return entities, entity_chunks


class ParsingTree():
    def __init__(self, node_list):
        self.id_to_node = []
        node_to_idx = {}
        entities, _ = get_entity(node_list)
        for idx, each_node in enumerate(node_list):
            curr_ent_idx = None
            for ent_idx, each_ent in enumerate(entities):
                if idx >= each_ent['span_bounds'][0] and idx < each_ent['span_bounds'][1]:
                    curr_ent_idx = ent_idx

            t_node = TreeNode(idx, each_node, ent_idx=curr_ent_idx)
            self.id_to_node.append(t_node)
            node_to_idx[each_node] = idx

        for node in self.id_to_node:
            for each in node.init_token.children:
                child_id = node_to_idx[each]
                node.children_id.append(child_id)
                node.children_token.append(self.id_to_node[child_id])
            if node.dep == 'ROOT':
                self.root = node
            else:
                parent_id = node_to_idx[node.init_token.head]
                self.parent_id = parent_id
                node.parent_token = self.id_to_node[parent_id]

            node.init_token = None

    def get_attribute(self):
        att_pairs = []
        for node in self.id_to_node:
            if node.pos == 'NOUN':
                for each_c in node.children_token:
                    if each_c.dep == 'amod' and each_c.pos == 'ADJ' or each_c.dep == 'compound':
                        att_pairs.append([each_c, node])
        return att_pairs

    def get_spo(self):
        sub, pred, obj = None, None, None
        if self.root.pos == 'VERB' or self.root.pos == 'AUX':
            #find the sub/obj
            sub = bfs(self.root, is_sub)
            obj = bfs(self.root, is_obj)
            
            last_obj = None
            if obj is not None:
                for each_chld in obj.children_token:
                    last_obj = bfs(each_chld, is_obj)

            if last_obj is not None:
                obj = [obj, last_obj]
            else:
                obj = get_all_dep_ent(obj, degree=1)

            pred = self.root
            if self.root.pos == 'AUX':
                pred_cat = [self.root]
                for each in pred.children_token:
                    if each.dep == 'prep':
                        pred_cat.append(each)
                pred = sorted(pred_cat, key=get_node_id)
            else:
                pred = get_all_dep_pred(pred, degree=1)
                
        elif self.root.pos == 'NOUN':
            # subject prep obj
            sub = self.root
            obj = bfs(self.root, is_obj) 

            if obj is not None:
                pred = [obj.parent_token]
            
                last_obj = None
                for each_chld in obj.children_token:
                    last_obj = bfs(each_chld, is_obj)

                if last_obj is not None:
                    obj = [obj, last_obj]
                else:
                    obj = get_all_dep_ent(obj, degree=1)


        return get_all_dep_ent(sub, degree=1), pred, obj
    

    def traverse(self):
        q = queue.Queue()
        q.put(self.root)
        while q.qsize() > 0:
            curr = q.get()
            print('curr')
            print(curr)

            print('parent')
            print(curr.parent_token)
            print('children')
            for each in curr.children_token:
                print(each)
                q.put(each)
            print()


class TreeNodeDict(dict):
    def __init__(self, idx, curr, parent_id=None, children_id=[]):

        super(TreeNodeDict, self).__init__()
        if isinstance(curr, dict):
            self['parent_id'] = None
            self.update(curr)
        else:
            self['id'] = idx
            self['text'] = curr.text
            self['dep'] = curr.dep
            self['pos'] = curr.pos
            if curr.parent_token is not None:
                self['parent_id'] = curr.parent_token.id
            else:
                self['parent_id'] = None

            self['ent_idx'] = curr.ent_idx
            self['rel_idx'] = curr.rel_idx

            self['children_id'] = children_id

    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value


class ParsingTreeDict():
    def __init__(self, parse_tree):
        self.id_to_node = dict()
        if isinstance(parse_tree, ParsingTree):
            node_list = parse_tree.id_to_node
            for idx, each_node in enumerate(node_list):
                # print(each_node)
                t_node = TreeNodeDict(idx, each_node, 
                                      parent_id=each_node.parent_id,
                                      hildren_id=each_node.children_id)
                self.id_to_node[t_node.id] = t_node
                self.root = TreeNodeDict(
                    parse_tree.root.id, 
                    parse_tree.root,
                    parse_tree.root.parent_id,
                    parse_tree.root.children_id,
                )

        elif isinstance(parse_tree, dict):
            self.id_to_node = {
                v['id']: TreeNodeDict(v['id'], v) 
                for k, v in parse_tree['id_to_node'].items()
            }
            
            self.root = TreeNodeDict(idx=None, 
                                     curr=parse_tree['root'])

    def bfs(self, start_node, match_cond, depth=-1, node_id=-1):
        q = queue.Queue()
        q.put(start_node)
        curr_depth = 0
        while q.qsize() > 0:
            curr = q.get()
            if match_cond(curr):
                return curr
            
            if depth > 0:
                if curr_depth >= depth:
                    q = queue.Queue()
                    continue

            curr_depth += 1
            for each in curr.children_id:
                if node_id > 0:
                    if each < node_id:
                        continue
                q.put(self.id_to_node[each])

    def get_all_dep_ent(self, start_node, degree=-1):
        if start_node is None:
            return None
        
        all_children = []
        q = queue.Queue()
        q.put(start_node)
        curr_deg = 0
        while q.qsize() > 0:
            curr = q.get()
            curr_deg += 1
            for each_cld in curr.children_id:
                each_cld = self.id_to_node[each_cld]

                if each_cld.dep != 'prep' and each_cld.dep != 'acl' and each_cld.dep != 'amod' and 'obj' not in each_cld.dep and 'sub' not in each_cld.dep :
                    if curr_deg >= degree and degree > 0:
                        q.put(each_cld)
                    if each_cld.dep != 'punct':
                        all_children.append(each_cld)
        all_children.append(start_node)
        all_children = sorted(all_children, key=get_node_id)
        return all_children

    def get_all_dep_pred(self, start_node, degree=-1):
        if start_node is None:
            return None
        
        all_children = []
        q = queue.Queue()
        q.put(start_node)
        curr_deg = 0
        while q.qsize() > 0:
            curr = q.get()
            curr_deg += 1
            for each_cld in curr.children_id:
                each_cld = self.id_to_node[each_cld]

                if each_cld.pos == 'ADP' or ('obj' not in each_cld.dep and 'sub' not in each_cld.dep) and each_cld.dep != 'prep':
                    if curr_deg < degree and degree > 0:
                        q.put(each_cld)
                    if each_cld.dep != 'punct':
                        all_children.append(each_cld)
        all_children.append(start_node)
        all_children = sorted(all_children, key=get_node_id)
        return all_children

    def get_all_dep_aux_pred(self, start_node, degree=-1):
        if start_node is None:
            return None
        
        all_children = []
        q = queue.Queue()
        q.put(start_node)
        curr_deg = 0
        while q.qsize() > 0:
            curr = q.get()
            curr_deg += 1
            for each_cld in curr.children_id:
                each_cld = self.id_to_node[each_cld]
                if each_cld.dep == 'prep':
                    if curr_deg < degree and degree > 0:
                        q.put(each_cld)
                    if each_cld.dep != 'punct':
                        all_children.append(each_cld)
        all_children.append(start_node)
        all_children = sorted(all_children, key=get_node_id)
        return all_children
    

    def get_spo(self):
        sub, pred, obj = None, None, None
        if self.root.pos == 'VERB':
            #find the sub/obj
            sub = self.bfs(self.root, is_sub)
            obj = self.bfs(self.root, is_obj)
            pred = self.root
            
            last_obj = None
            if obj is not None:
                for cld_id in obj.children_id:
                    each_chld = self.id_to_node[cld_id]
                    last_obj = self.bfs(each_chld, is_obj)

            if last_obj is not None:
                obj = [obj, last_obj]
            else:
                obj = self.get_all_dep_ent(obj, degree=1)

            pred = self.get_all_dep_pred(pred, degree=1)

            sub = self.get_all_dep_ent(sub, degree=1)

        elif self.root.pos == 'AUX':
            # import ipdb; ipdb.set_trace()
            sub = self.bfs(self.root, is_sub)
            obj = self.bfs(self.root, is_obj_only, node_id=self.root.id)
            if obj is None:
                obj = self.bfs(self.root, is_obj, node_id=self.root.id)
                if obj is not None:
                    obj = [obj]
            else:
                obj = self.get_all_dep_ent(obj, degree=1)
                

            pred = self.get_all_dep_aux_pred(self.root, degree=1)
            sub = self.get_all_dep_ent(sub, degree=1)

                
        elif self.root.pos == 'NOUN':
            # subject prep obj
            sub = self.root
            obj = self.bfs(self.root, is_obj) 

            if obj is not None:
                pred = [self.id_to_node[obj.parent_id]]

                last_obj = None
                for cld_id in obj.children_id:
                    each_chld = self.id_to_node[cld_id]
                    last_obj = self.bfs(each_chld, is_obj)

                if last_obj is not None:
                    obj = [obj, last_obj]
                else:
                    obj = self.get_all_dep_ent(obj, degree=1)
            sub = self.get_all_dep_ent(sub, degree=1)

        return sub, pred, obj
    

    def traverse(self):
        q = queue.Queue()
        q.put(self.root)
        while q.qsize() > 0:
            curr = q.get()
            print('curr')
            print(curr)

            print('parent')
            print(curr.parent_token)
            print('children')
            for each in curr.children_token:
                print(each)
                q.put(each)
            print()



def parsing_complete(data_path, data_name):
    print(data_path)

    with open(f"{data_path}/{data_name}.json", 'r') as file:
        anno = json.load(file)

    nlp_spacy = spacy.load("en_core_web_sm")

    ent_cnter = Counter()
    pred_cnter = Counter()
    noun_cnter = Counter()
    verb_cnter = Counter()
    attr_cnter = Counter()

    anno_all = defaultdict(list)

    for each_anno in tqdm(anno):
        if 'image_id' not in each_anno:
            each_anno['image_id'] = each_anno['image']
        anno_all[each_anno['image_id']].append(each_anno)

    for anno_id in tqdm(list(anno_all.keys())):
        each_anno = anno_all[anno_id]
        for cap_idx, each_cap in enumerate(each_anno):
            try:
                if type(each_cap['caption']) == list: each_cap['caption'] = each_cap['caption'][0] # TODO: temp approximation
                phrase = each_cap['caption'] 

                if phrase[:len('there is')] == 'there is':
                    phrase = phrase[len('there is'):]
                if phrase[:len('there are')] == 'there are':
                    phrase = phrase[len('there are'):]

                doc = nlp_spacy(phrase)
                parse_tree = ParsingTree(doc)

                # phrase = 'white monitor is on'
                doc = nlp_spacy(phrase)
                entities, entity_chunks = get_entity(doc)

                for each_ent in entity_chunks:
                    ent_cnter[str(each_ent)] += 1

                ent_pos = [(each['span_bounds'][0], each['span_bounds'][1]) for each in entities]

                noun_pos = []
                verb_pos = []
                for node in parse_tree.id_to_node:
                    if node.pos == 'NOUN':
                        noun_pos.append((node.id, node.id + 1))
                        noun_cnter[node.text] += 1
                    elif node.pos == 'VERB':
                        verb_pos.append((node.id, node.id + 1))
                        verb_cnter[node.text] += 1

                attr_pos = []
                for attr_pair in parse_tree.get_attribute():
                    attr_pos.append((attr_pair[0].id, attr_pair[0].id + 1))
                    attr_cnter[attr_pair[0].text] += 1

                rel_pos = []
                # spo_pos = []
                for ent_idx in range(len(entities) - 1):
                    relation = []
                    for t_idx in range(entities[ent_idx]['span_bounds'][-1], entities[ent_idx + 1]['span_bounds'][0]):
                        relation.append(parse_tree.id_to_node[t_idx])

                    text = ' '.join([each.text for each in relation]).strip('.').strip(',').strip('-').strip(' ')
                    if 'and' not in text and text != 'of' and text != 'is' and len(text) > 0:
                        pred_cnter[text] += 1
                        # print(text)
                        rel_pos.append((entities[ent_idx]['span_bounds'][-1], entities[ent_idx + 1]['span_bounds'][0]))

                parse_res = {
                    'root_pos': parse_tree.root.pos,
                    'parsed_text': [parse_tree.id_to_node[each].text for each in range(len(parse_tree.id_to_node))],
                    'parsed_pos': {'ent': ent_pos,
                                   'pred': rel_pos,
                                   'noun': noun_pos,
                                   'verb': verb_pos,
                                   'attr': attr_pos},
                }
                # print(parse_res)
                each_cap['parse_res'] = parse_res

            except Exception as error:
                print("An error occurred:", type(error).__name__, "-", error)
                print(anno_id, cap_idx, each_cap)
    
    with open(f"{data_path}/{data_name}_w_parsing_nva.json", 'w') as file:
        json.dump(anno, file)

    with open(f"{data_path}/{data_name}_cnter_nva.json", 'w') as file:
        json.dump({'ent': ent_cnter, 'pred': pred_cnter, 'noun': noun_cnter, 'verb': verb_cnter, 'attr': attr_cnter},
                  file)
        
    print('ICCC data annotation dir:', f"{data_path}/{data_name}_w_parsing_nva.json")
    print('ICCC data concept base dir:', f"{data_path}/{data_name}_cnter_nva.json")


if __name__ == "__main__":
    data_path = sys.argv[1] #'data/coco/annotations'
    data_name = sys.argv[2] #'coco_karpathy_val'
    parsing_complete(data_path, data_name)