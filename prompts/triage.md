Classify one Australian Securities Exchange announcement into exactly one class:

- `skip`: a cleansing notice, trading halt without substantive information, or
  administrative correction that adds no facts or useful event details.
- `form`: a structured filing such as an Appendix 2A/3B, director-interest
  notice, cessation-of-securities notice, or substantial-holder notice.
- `full`: an announcement with substantive company, project, financial,
  operational, resource, reserve, drilling, funding, transaction, or guidance
  information. Meeting notices, AGM venue/time announcements, investor
  webcast notices, and small-share-sale-facility announcements are `full`
  because their event details may be asked about later, even when they include
  a proxy form or standardized attachment.

Use the headline and metadata as evidence, not as rules. Price-sensitive
announcements and quarterly or half/full-year reports should normally be
`full`. A standard form remains `form` even when its values are important;
Wave 2 will process it with a dedicated schema.
Use `form` only for prescribed, standardized filings. A facility, transaction,
or funding announcement that explains terms or company impact is `full`.
Do not classify an AGM or webcast notice as a `form` merely because it includes
a proxy form or is primarily informational.

Return JSON matching the required schema:

- `class`: exactly `skip`, `form`, or `full`
- `form_type`: for `form`, choose `appendix_2a`, `appendix_3b`,
  `director_interest`, `substantial_holder`, or `other` when none of those
  types applies; otherwise `null`. Always include this property.
- `reason`: one concise sentence grounded in the supplied headline, metadata,
  or text excerpt

Do not invent facts, infer values absent from the excerpt, or return markdown.
