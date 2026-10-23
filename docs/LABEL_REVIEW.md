# Label review: real-regulation set

These labels were drafted with AI assistance and **no person has checked them yet**. Until you do,
every number on the `real` suite is provisional (see [EVAL_PLAN.md](EVAL_PLAN.md)).

**How to review.** For each row, read the requirement in `data/regulations_real/` and the clause in
`data/policies/`. If you agree, set `"status": "reviewed"` on that line of
`data/eval/labels_real.jsonl`. If you disagree, change `relevant`, `acceptable` or `impact` there and
say why in the commit message. Rows marked *Debatable* are where a second opinion matters most.

- **Relevant:** the clause(s) the report must flag. The first one decides the impact label.
- **Also acceptable:** clauses that are fine to flag but not required.

## CERT-In Directions under Section 70B(6) of the IT Act, 2000 (28 April 2022)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Synchronisation of system clocks | Missing from policy | `info-security§5.2` Log Retention | `info-security§1.1` Scope, `info-security§5.1` Security Incident Reporting | No policy mentions clock synchronisation. Closest clause is log retention, since synchronised clocks matter for logs. | ☐ |
| R2 | Reporting of cyber incidents within 6 hours | Needs change | `info-security§5.1` Security Incident Reporting | `data-privacy§5.1` Personal Data Breach Notification | Policy reports cyber incidents within 24 hours; the direction requires 6 hours. Data breaches are also Annexure I incidents, so the breach clause is acceptable. | ☐ |
| R3 | Point of Contact for CERT-In | Missing from policy | `info-security§5.1` Security Incident Reporting | — | No designated CERT-In Point of Contact anywhere; the incident-reporting clause is where it belongs. | ☐ |
| R4 | Logs kept for 180 days within India | Needs change | `info-security§5.2` Log Retention | `data-privacy§4.1` Data Storage Location, `data-privacy§3.2` Data Retention | Logs are kept 90 days, not 180, and the policy says nothing about keeping them in India (data is stored in Singapore and the US). | ☐ |
| R5 | Subscriber records of hosting and VPN providers | Does not apply | — | — | Applies to data centres, VPS, cloud and VPN providers. Northwind Pay is a payments wallet, not one of these. | ☐ |
| R6 | KYC records of virtual asset service providers | Does not apply | — | — | Applies to virtual asset service providers and exchanges. Northwind Pay does not deal in virtual assets, though its KYC record-keeping clause looks similar. | ☐ |

## Digital Personal Data Protection Act, 2023 (No. 22 of 2023)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Free, specific and informed consent (Section 6(1)) | Needs change | `data-privacy§2.1` Lawful Basis and Consent | — | A single checkbox bundles account operation, analytics and marketing, and refusing it blocks the account: not free or specific consent. | ☐ |
| R2 | Consent requests in plain language with a contact (Section 6(3)) | Missing from policy | `data-privacy§2.1` Lawful Basis and Consent | `data-privacy§6.2` Data Protection Officer | The consent clause says nothing about plain language, language choice or showing a contact in the consent request. | ☐ |
| R3 | Withdrawal as easy as giving consent (Section 6(4)) | Needs change | `data-privacy§2.2` Withdrawal of Consent | — | Consent is given with one checkbox but withdrawn only by emailing support: not comparably easy. | ☐ |
| R4 | Data Processors only under a valid contract (Section 8(2)) | Already compliant | `data-privacy§4.2` Third-Party Sharing | `data-privacy§1.1` Purpose and Scope, `info-security§6.1` Vendor Security | Sharing with partners already requires a written contract. Debatable: partners may be other fiduciaries rather than processors. | ☐ |
| R5 | Intimation of every personal data breach (Section 8(6)) | Needs change | `data-privacy§5.1` Personal Data Breach Notification | `data-privacy§5.2` Breach Register | Customers are told only when the breach is high risk; the Act requires telling each affected person. | ☐ |
| R6 | Erasure once the purpose is served (Section 8(7)) | Needs change | `data-privacy§3.2` Data Retention | `data-privacy§3.3` Deletion Requests, `kyc-aml§6.1` Record Keeping | Blanket 10-year retention after closure, rather than erasure once the purpose is served (unless another law requires it). | ☐ |
| R7 | Published contact for data questions (Section 8(9)) | Already compliant | `data-privacy§6.2` Data Protection Officer | — | The DPO's contact details are already published on the website. | ☐ |
| R8 | Grievance redressal for personal data (Section 13) | Missing from policy | `customer-grievance§1.1` Scope | `data-privacy§6.2` Data Protection Officer, `customer-grievance§3.1` Resolution Timeline | The grievance policy covers complaints in general but not personal-data grievances. Debatable: a reviewer may read it as already covering them. | ☐ |
| R9 | Verifiable parental consent for children (Section 9(1)) | Needs change | `data-privacy§6.1` Children's Data | — | A child is under 18 under the Act; the policy takes no parental consent for users aged 16 to 18. | ☐ |
| R10 | No tracking or targeted advertising of children (Section 9(3)) | Missing from policy | `data-privacy§6.1` Children's Data | `data-privacy§2.1` Lawful Basis and Consent | Nothing forbids tracking or targeted advertising of under-18 users, and the bundled consent includes marketing and analytics. | ☐ |
| R11 | Erasure on request (Section 12(3)) | Already compliant | `data-privacy§3.3` Deletion Requests | `data-privacy§3.2` Data Retention | Deletion requests are honoured within 45 days except where law requires retention. The Act sets no deadline. | ☐ |

## RBI Circular — Customer Protection: Limiting Liability of Customers in Unauthorised Electronic Banking Transactions (6 July 2017)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Mandatory SMS alerts | Needs change | `payments-ops§4.1` Transaction Alerts | — | SMS alerts only for debits above 500; the circular requires them for every electronic transaction. | ☐ |
| R2 | Reporting channels available 24x7 | Needs change | `payments-ops§3.2` Reporting Unauthorised Transactions | `customer-grievance§2.1` Complaint Channels | Reporting only through the app or helpline (and the helpline runs 9-6 on working days), not 24x7 through multiple channels. | ☐ |
| R3 | Immediate acknowledgement with a complaint number | Needs change | `customer-grievance§2.2` Acknowledgement | `payments-ops§3.2` Reporting Unauthorised Transactions | Complaints are acknowledged within 5 working days; fraud reports need an immediate acknowledgement with a complaint number. | ☐ |
| R4 | Blocking further unauthorised transactions | Missing from policy | `payments-ops§3.2` Reporting Unauthorised Transactions | `payments-ops§3.1` Fraud Monitoring | Nothing says the account is blocked or protected immediately after a fraud report. | ☐ |
| R5 | Zero liability when reported within three working days | Needs change | `payments-ops§3.3` Customer Liability for Unauthorised Transactions | — | Liability is capped at 10,000 within 7 days, never zero; the circular gives zero liability within 3 working days. | ☐ |
| R6 | Capped liability for reports within four to seven working days | Needs change | `payments-ops§3.3` Customer Liability for Unauthorised Transactions | — | After 7 days the customer bears the full loss, and the 4-7 day cap is not tied to the transaction value. | ☐ |

## RBI Circular — Harmonisation of Turn Around Time (TAT) and customer compensation for failed transactions using authorised Payment Systems (20 September 2019)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Compensation without waiting for a complaint | Needs change | `customer-grievance§4.2` Compensation | `customer-grievance§4.1` Failed Transaction Refunds | The policy pays no compensation at all; the circular requires automatic compensation. | ☐ |
| R2 | Reversal of failed transfers by T + 1 day | Needs change | `customer-grievance§4.1` Failed Transaction Refunds | `customer-grievance§4.2` Compensation | Refund within 10 working days after the customer complains, instead of automatic reversal by T + 1 day. | ☐ |
| R3 | Cash not dispensed at an ATM | Does not apply | — | — | Applies to ATMs and micro-ATMs. Northwind Pay operates a wallet and runs no ATMs, though its failed-refund clause looks similar. | ☐ |

## RBI Master Direction on Digital Payment Security Controls (18 February 2021)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Multi-factor authentication for payments | Needs change | `info-security§2.2` Customer Authentication for Payments | — | OTP only above 5,000; the direction requires multi-factor authentication for payments in general. | ☐ |
| R2 | Alerts for every payment and account change | Needs change | `payments-ops§4.1` Transaction Alerts | `info-security§2.2` Customer Authentication for Payments | Alerts only for debits above 500; no alerts for credits, beneficiary changes or limit changes. | ☐ |
| R3 | Blocking after failed log-in attempts | Missing from policy | `info-security§2.2` Customer Authentication for Payments | `info-security§2.1` Employee Authentication | No limit on failed log-in attempts and no notification of them. | ☐ |
| R4 | Marking a transaction as fraudulent in the app | Already compliant | `payments-ops§3.2` Reporting Unauthorised Transactions | — | Customers can already report unauthorised transactions through the app. | ☐ |
| R5 | Breach monitoring around the clock | Needs change | `payments-ops§3.1` Fraud Monitoring | `info-security§5.1` Security Incident Reporting | Fraud rules run on the previous day's transactions, not around the clock. | ☐ |
| R6 | Reconciliation within 24 hours | Missing from policy | `payments-ops§4.2` Merchant Settlement | `payments-ops§3.1` Fraud Monitoring | No reconciliation framework at all; settlement timing is the closest clause. Debatable choice of closest clause. | ☐ |
| R7 | No card details in plain text | Already compliant | `info-security§3.2` Card Data Storage | `info-security§3.1` Encryption | Card numbers are stored encrypted, so not in plain text. (A tokenisation rule would conflict; this one does not.) | ☐ |

## RBI Master Direction — Know Your Customer (KYC) Direction, 2016 (as updated)

| Req | Requirement | Expected | Relevant | Also acceptable | Why | ✓ |
|---|---|---|---|---|---|---|
| R1 | Periodicity of KYC updation | Already compliant | `kyc-aml§3.1` Periodic KYC Updation | — | The policy already uses exactly 2 / 8 / 10 years. | ☐ |
| R2 | Advance notices before KYC falls due | Missing from policy | `kyc-aml§3.1` Periodic KYC Updation | — | No advance notices to customers before KYC falls due. | ☐ |
| R3 | V-CIP recordings stored in India | Needs change | `data-privacy§4.1` Data Storage Location, `kyc-aml§2.2` Video-based Customer Identification | — | Personal data is stored in Singapore and the US; V-CIP recordings must stay in India. The V-CIP clause itself is silent on location. | ☐ |
| R4 | Non-face-to-face customers treated as high risk | Missing from policy | `kyc-aml§3.2` Risk Categorisation | `kyc-aml§2.1` Customer Identification | Risk categories do not say non-face-to-face customers start as high risk. | ☐ |
| R5 | Preservation of records for five years | Already compliant | `kyc-aml§6.1` Record Keeping | — | Records are already kept 5 years after the relationship ends or the transaction completes. | ☐ |
