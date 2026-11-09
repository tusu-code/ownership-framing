import json, re, pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
O='wildchat/out/'
c=pd.read_json(O+'candidates.jsonl',lines=True).set_index('h')
L=lambda f: pd.DataFrame([json.loads(l) for l in open(O+f)]).query('ok').drop_duplicates('h',keep='last').set_index('h')
d,h=L('classified.jsonl'),L('classified_haiku.jsonl')
union=sorted(set(d[d.assess==True].index)|set(h[h.assess==True].index))
rng=np.random.default_rng(20261108); order=list(rng.permutation(union))
fin=pd.read_csv(O+'assess_final.csv',index_col=0)
key=pd.DataFrame({'item':range(1,len(order)+1),'h':order})
key['model_assess']=key.h.isin(fin.index); key['model_relation']=key.h.map(fin.relation).fillna('')
key.to_csv('wildchat/audit/key_PRIVATE.csv',index=False)
clean=lambda s: ILLEGAL_CHARACTERS_RE.sub('',s)
wb=Workbook(); ws=wb.active; ws.title='audit'
ws.append(['item','user message (opening; [...] marks a cut)','assess (Y/N)','relation (own / reviewer / other / none)','note'])
for i,hh in enumerate(order,1):
    r=c.loc[hh]; txt=r['head'][:2500]+('\n[...]\n'+r['tail'][-800:] if r['tail'] else '')
    ws.append([i,clean(txt),'','',''])
for col,w in zip('ABCDE',(6,110,12,22,30)): ws.column_dimensions[col].width=w
for row in ws.iter_rows():
    for x in row: x.alignment=Alignment(wrap_text=True,vertical='top')
for x in ws[1]: x.font=Font(bold=True); x.fill=PatternFill('solid',fgColor='D9E9EC')
dv1=DataValidation(type='list',formula1='"Y,N"',allow_blank=True); dv2=DataValidation(type='list',formula1='"own,reviewer,other,none"',allow_blank=True)
ws.add_data_validation(dv1); ws.add_data_validation(dv2); dv1.add(f'C2:C{len(order)+1}'); dv2.add(f'D2:D{len(order)+1}')
ws.freeze_panes='C2'
g2=wb.create_sheet('instructions'); g2.column_dimensions['A'].width=140
for line in ["Read each user message and fill two columns. Do not look at any model labels.",
 "assess = Y if the user asks the assistant to judge the quality or merit of a scholarly manuscript (paper, thesis, dissertation, proposal, or a section or abstract of one): review, evaluate, critique, give feedback, rate, or say whether it is good, ready, publishable. N for writing, rewriting, polishing, translating, summarising, citing or formatting, school essays, stories, etc.",
 "relation (only if assess = Y): own = the user or their team wrote it; reviewer = the user is reviewing or refereeing it for a journal, conference, editor or committee; other = someone else wrote it and the user is not reviewing it; none = no relation stated or implied.",
 "Leave relation empty when assess = N. Use 'note' for doubts.",
 f"{len(order)} items in random order; about 30-60 minutes."]:
    g2.append([line]); g2.cell(g2.max_row,1).alignment=Alignment(wrap_text=True)
wb.save('wildchat/audit/wildchat_audit.xlsx'); print(len(order))
