# Upgrade 2.0
The results are promising. However, a new massive upgrade is successfully verified and requires careful implementation. Therefore, study the notebook and other files in https://github.com/abdoljh/OCR-me/tree/main/results and implement the following tasks:
	**1.	Real books simply mean large memory usage.** Monitoring memory, in Streamlit environments, becomes a must. As a suggestion: if the estimated size of generated files would exceed a specific limit, split the source pdf into more than one. Make use of File_Sizes.txt in estimation.
	**2.	Modes of operation become four**, arranged according to their importance:
(a) The mental model (single-book workflow): one source->multi outputs 
(b) Batch processing (multi-pdf workflow): multiple sources->multi outputs
(c) A small caveat (side product): `page_export_v2.py`, which doesn’t validate stripping pdf first, renders every full page at 400 DPI — no crops, no error.
(d) The visual mode (the originally implemented solution).
These modes should be arranged as shown starting with the mental model.
	**3.	Auto-detect margin should be Off in default**. No display of any page unless the visual mode is selected. This would help in preserving memory.
	**4.	The main target of first priority is mode 1 and its results**: stripped pdf, page images, footer images. The last two would be the inputs to the following OCR.
