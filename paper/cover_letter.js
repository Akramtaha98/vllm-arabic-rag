const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
} = require("docx");

const bodyPara = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, size: 22, ...opts })],
  spacing: { after: 160 },
  alignment: AlignmentType.LEFT,
});

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1350, right: 1350 } },
    },
    children: [
      new Paragraph({ children: [new TextRun({ text: "Akram Taha", bold: true, size: 22 })], spacing: { after: 0 } }),
      new Paragraph({ children: [new TextRun({ text: "College of Computer Engineering, University of Technology - Iraq, Baghdad, Iraq", size: 20 })], spacing: { after: 0 } }),
      new Paragraph({ children: [new TextRun({ text: "Center for Artificial Intelligence Technology (CAIT), FTSM, Universiti Kebangsaan Malaysia (UKM), Bangi, Selangor, Malaysia", size: 20 })], spacing: { after: 0 } }),
      new Paragraph({ children: [new TextRun({ text: "Email: akramtaha30@gmail.com  |  ORCID: 0009-0002-4020-8060", size: 20 })], spacing: { after: 240 } }),

      new Paragraph({ children: [new TextRun({ text: new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }), size: 20 })], spacing: { after: 240 } }),

      new Paragraph({ children: [new TextRun({ text: "Dear Editor-in-Chief,", size: 22 })], spacing: { after: 160 } }),

      bodyPara(
        "I am submitting the manuscript “Semantic-Driven Context Pruning for Arabic RAG Systems: Toward Memory-Efficient vLLM-Based Deployment” for consideration as a research article in Applied Intelligence. The manuscript has one author (myself), has not been published, and is not under consideration elsewhere."
      ),

      bodyPara(
        "The paper addresses a memory bottleneck specific to Arabic Retrieval-Augmented Generation (RAG): Arabic's morphological richness inflates token counts relative to English, increasing the Key-Value cache that memory-efficient serving engines such as vLLM must maintain per request. I introduce a Lightweight Semantic Pruning Middleware (LSPM) that prunes retrieved passages at the sentence level with a cross-encoder before generation. In a properly powered re-test on 140 real questions from the Arabic Reading Comprehension Dataset (980 live model generations), LSPM significantly outperforms naive length-matched truncation at the most aggressive compression ratio tested and is statistically equivalent to the unpruned answer, a result I report honestly alongside the ratios where the comparison remains open rather than as a uniform claim."
      ),

      bodyPara(
        "The manuscript is explicit about its scope: it is an architecture-and-preliminary-validation study. No GPU or vLLM server was available for this work, so I report an analytical, clearly labeled projection of KV-cache savings rather than a measured throughput result, and I specify the benchmark protocol a follow-up study would need to measure it directly. vLLM is the paper's intended deployment target, not an evaluated one."
      ),

      bodyPara(
        "The complete implementation, raw experimental data, and analysis code will be made publicly available at https://github.com/Akramtaha98/vllm-arabic-rag, with a tagged release and a Zenodo-archived DOI, before final submission, to support independent reproduction of every reported result."
      ),

      bodyPara(
        "This work involved no human participants or personal data, and I have no conflicts of interest to disclose. Large language model assistance was used for implementation, literature verification, and drafting under my direction; this is disclosed in full in the manuscript, and I take full responsibility for the accuracy of all claims and results reported."
      ),

      bodyPara(
        "Thank you for considering this manuscript. I would be pleased to suggest reviewers or provide any further information the editorial process requires."
      ),

      new Paragraph({ children: [new TextRun({ text: "Sincerely,", size: 22 })], spacing: { after: 240 } }),
      new Paragraph({ children: [new TextRun({ text: "Akram Taha", size: 22 })], spacing: { after: 0 } }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/tmp/cover_letter.docx", buf);
  console.log("written");
});
