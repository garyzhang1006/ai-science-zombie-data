"""Genre corpus fetched from the live web, 2026-08-17.

Twelve papers from NeurIPS workshops and the peer-reviewed metascience
literature that this submission sits alongside. The abstracts below are
verbatim; they are the measurement target for tone and rhythm, replacing
the earlier hand-asserted style targets.

Selection rule: a paper qualifies if it (a) appeared at a NeurIPS workshop
or an adjacent venue, and (b) makes an empirical or position claim about
scientific practice rather than about a scientific domain. That is the
genre this submission competes in.

Only papers whose abstract was retrieved verbatim carry text and count
toward the measured profile. The rest are recorded for provenance.
"""

CORPUS = [
    dict(
        key="kobak2025delving",
        title="Delving into LLM-assisted writing in biomedical publications "
              "through excess vocabulary",
        venue="Science Advances 11(27):eadt3813, 2025",
        url="https://arxiv.org/abs/2406.07016",
        why="the method this submission critiques in Sec 4.3",
        text="""Large language models (LLMs) like ChatGPT can generate and revise
text with human-level performance. These models come with clear limitations: they
can produce inaccurate information, reinforce existing biases, and be easily
misused. Yet, many scientists use them for their scholarly writing. But how
wide-spread is such LLM usage in the academic literature? To answer this question
for the field of biomedical research, we present an unbiased, large-scale
approach: we study vocabulary changes in over 15 million biomedical abstracts
from 2010-2024 indexed by PubMed, and show how the appearance of LLMs led to an
abrupt increase in the frequency of certain style words. This excess word
analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs.
This lower bound differed across disciplines, countries, and journals, reaching
40% for some subcorpora. We show that LLMs have had an unprecedented impact on
scientific writing in biomedical research, surpassing the effect of major world
events such as the Covid pandemic.""",
    ),
    dict(
        key="rigour2024",
        title="On the Rigour of Scientific Writing: Criteria, Analysis, and Insights",
        venue="arXiv:2410.04981",
        url="https://arxiv.org/abs/2410.04981",
        why="computational measurement of a property of scientific writing",
        text="""Rigour is crucial for scientific research as it ensures the
reproducibility and validity of results and findings. Despite its importance,
little work exists on modelling rigour computationally, and there is a lack of
analysis on whether these criteria can effectively signal or measure the rigour
of scientific papers in practice. In this paper, we introduce a bottom-up,
data-driven framework to automatically identify and define rigour criteria and
assess their relevance in scientific writing. Our framework includes rigour
keyword extraction, detailed rigour definition generation, and salient criteria
identification. Furthermore, our framework is domain-agnostic and can be tailored
to the evaluation of scientific rigour for different areas, accommodating the
distinct salient criteria across fields. We conducted comprehensive experiments
based on datasets collected from two high impact venues for Machine Learning and
NLP (i.e., ICLR and ACL) to demonstrate the effectiveness of our framework in
modelling rigour. In addition, we analyse linguistic patterns of rigour,
revealing that framing certainty is crucial for enhancing the perception of
scientific rigour, while suggestion certainty and probability uncertainty
diminish it.""",
    ),
    dict(
        key="ethics2021",
        title="AI Ethics Statements: Analysis and lessons learnt from NeurIPS "
              "Broader Impact Statements",
        venue="arXiv:2111.01705",
        url="https://arxiv.org/abs/2111.01705",
        why="empirical metascience on NeurIPS itself, releases its dataset",
        text="""Ethics statements have been proposed as a mechanism to increase
transparency and promote reflection on the societal impacts of published
research. In 2020, the machine learning (ML) conference NeurIPS broke new ground
by requiring that all papers include a broader impact statement. This requirement
was removed in 2021, in favour of a checklist approach. The 2020 statements
therefore provide a unique opportunity to learn from the broader impact
experiment: to investigate the benefits and challenges of this and similar
governance mechanisms, as well as providing an insight into how ML researchers
think about the societal impacts of their own work. Such learning is needed as
NeurIPS and other venues continue to question and adapt their policies. To enable
this, we have created a dataset containing the impact statements from all NeurIPS
2020 papers, along with additional information such as affiliation type, location
and subject area, and a simple visualisation tool for exploration. We also
provide an initial quantitative analysis of the dataset, covering
representation, engagement, common themes, and willingness to discuss potential
harms alongside benefits. We investigate how these vary by geography, affiliation
type and subject area. Drawing on these findings, we discuss the potential
benefits and negative outcomes of ethics statement requirements, and their
possible causes and associated challenges. These lead us to several lessons to be
learnt from the 2020 requirement: (i) the importance of creating the right
incentives, (ii) the need for clear expectations and guidance, and (iii) the
importance of transparency and constructive deliberation. We encourage other
researchers to use our dataset to provide additional analysis, to further our
understanding of how researchers responded to this requirement, and to
investigate the benefits and challenges of this and related mechanisms.""",
    ),
    dict(
        key="negresults2024",
        title="Position: Embracing Negative Results in Machine Learning",
        venue="arXiv:2406.03980",
        url="https://arxiv.org/abs/2406.03980",
        why="the closest genre-mate for how to frame a null as the contribution",
        text="""Publications proposing novel machine learning methods are often
primarily rated by exhibited predictive performance on selected problems. In this
position paper we argue that predictive performance alone is not a good indicator
for the worth of a publication. Using it as such even fosters problems like
inefficiencies of the machine learning research community as a whole and setting
wrong incentives for researchers. We therefore put out a call for the publication
of "negative" results, which can help alleviate some of these problems and
improve the scientific output of the machine learning research community. To
substantiate our position, we present the advantages of publishing negative
results and provide concrete measures for the community to move towards a
paradigm where their publication is normalized.""",
    ),
    dict(
        key="rnd2025",
        title="Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic "
              "Algorithm for Assessing Novelty",
        venue="arXiv:2503.01508",
        url="https://arxiv.org/abs/2503.01508",
        why="contrast case: an assertive-claims paper, shows the register we avoid",
        text="""In the pursuit of Artificial General Intelligence (AGI), automating
the generation and evaluation of novel research ideas is a key challenge in
AI-driven scientific discovery. This paper presents Relative Neighbor Density
(RND), a domain-agnostic algorithm for novelty assessment in research ideas that
overcomes the limitations of existing approaches by comparing an idea's local
density with its adjacent neighbors' densities. We first developed a scalable
methodology to create test set without expert labeling, addressing a fundamental
challenge in novelty assessment. Using these test sets, we demonstrate that our
RND algorithm achieves state-of-the-art (SOTA) performance in computer science
(AUROC=0.820) and biomedical research (AUROC=0.765) domains. Most significantly,
while SOTA models like Sonnet-3.7 and existing metrics show domain-specific
performance degradation, RND maintains consistent accuracies across domains by
its domain-invariant property, outperforming all benchmarks by a substantial
margin (0.795 v.s. 0.597) on cross-domain evaluation. These results validate RND
as a generalizable solution for automated novelty assessment in scientific
research.""",
    ),
    dict(
        key="agents4science2025",
        title="Exploring the use of AI authors and reviewers at Agents4Science",
        venue="arXiv:2511.15534",
        url="https://arxiv.org/abs/2511.15534",
        why="reports what a scholarly-practice experiment actually showed",
        text="""There is growing interest in using AI agents for scientific
research, yet fundamental questions remain about their capabilities as scientists
and reviewers. To explore these questions, we organized Agents4Science, the first
conference in which AI agents serve as both primary authors and reviewers, with
humans as co-authors and co-reviewers. Here, we discuss the key learnings from
the conference and their implications for human-AI collaboration in science.""",
    ),
    # Provenance-only: located and read, abstract not retrieved verbatim in full.
    dict(key="confmodel2025",
         title="Position: The Current AI Conference Model is Unsustainable!",
         venue="arXiv:2508.04586", url="https://arxiv.org/abs/2508.04586",
         why="position-paper rhetoric: legitimize, then diagnose", text=None),
    dict(key="repro2025",
         title="Reproducibility: The New Frontier in AI Governance",
         venue="arXiv:2510.11595", url="https://arxiv.org/abs/2510.11595",
         why="uses 'we posit' rather than asserting causation", text=None),
    dict(key="agentic2026",
         title="Agentic AI Scientists Are Not Built For Autonomous Scientific Discovery",
         venue="arXiv:2605.08956", url="https://arxiv.org/abs/2605.08956",
         why="negative finding stated in the title", text=None),
    dict(key="reprostd2026",
         title="NeurIPS Should Require Reproducibility Standards for Frontier AI "
               "Safety Claims",
         venue="arXiv:2605.08192", url="https://arxiv.org/abs/2605.08192",
         why="NeurIPS-venue metascience advocacy", text=None),
    dict(key="isotonic2026",
         title="Recommending Best Paper Awards for ML/AI Conferences via the "
               "Isotonic Mechanism",
         venue="arXiv:2601.15249", url="https://arxiv.org/abs/2601.15249",
         why="quantitative treatment of a peer-review process", text=None),
    dict(key="wikillm2025",
         title="Wikipedia in the Era of LLMs: Evolution and Risks",
         venue="arXiv:2503.02879", url="https://arxiv.org/abs/2503.02879",
         why="corpus-level measurement of LLM penetration into a public corpus",
         text=None),
]

WITH_TEXT = [c for c in CORPUS if c["text"]]
