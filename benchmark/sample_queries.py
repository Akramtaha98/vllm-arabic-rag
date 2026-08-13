"""
Shared Arabic sample-query pool and system prompt, used by locustfile.py,
request_manifest.py, and edge_gateway.py. Split out from locustfile.py so
that building a request manifest (a plain data/CPU task) does not require
importing locust (which pulls in gevent and, in some environments, fails
to import for unrelated dependency reasons -- e.g. missing zope.event --
that have nothing to do with query-manifest generation).
"""

SAMPLE_QUERIES = [
    "متى تأسست جامعة الملك سعود؟",
    "ما هي اهتمامات قسم الحاسب في الجامعة؟",
    "كيف يكون الطقس في الرياض خلال الصيف؟",
    "ماذا يوجد في مكتبة الجامعة؟",
    "ما هو مركز أبحاث الذكاء الاصطناعي؟",
]

SYSTEM_PROMPT = "أنت مساعد ذكي تجيب بدقة بالاعتماد فقط على السياق المتاح لك وبشكل مختصر."
