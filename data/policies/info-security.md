# Northwind Pay — Information Security Policy

Version 4.2 · Owner: Chief Information Security Officer · Fictional sample document for demonstration only.

## 1.1 Scope

This policy applies to all information systems, networks, applications and data owned or operated by Northwind Pay, and to all employees and vendors who access them.

## 2.1 Employee Authentication

Employee passwords must be at least 8 characters long and changed every 90 days. Multi-factor authentication (MFA) is mandatory for administrator accounts only.

## 2.2 Customer Authentication for Payments

Customers authenticate with their app PIN. A one-time password (OTP) sent by SMS is required as an additional factor only for transactions above 5,000.

## 3.1 Encryption

Data at rest is encrypted with AES-256. Data in transit is protected with TLS 1.2 or higher. Encryption keys are stored in a hardware security module and rotated every year.

## 3.2 Card Data Storage

To enable one-click repeat payments, full card numbers are stored in encrypted form on our servers along with the expiry date. Access to stored card data is limited to the payments engineering team.

## 4.1 Access Reviews

User access rights to production systems are reviewed annually by system owners. Access that is no longer required is revoked within 10 working days of the review.

## 4.2 Privileged Access

Privileged (administrator) accounts are granted only with written approval from the CISO. Privileged sessions on production systems are logged.

## 5.1 Security Incident Reporting

Cyber security incidents are reported to the national cyber security authority within 24 hours of detection. The incident response team leads containment and root-cause analysis.

## 5.2 Log Retention

System and application logs are retained for ninety (90) days on the central logging platform and then deleted.

## 6.1 Vendor Security

Vendors with access to Northwind Pay systems or data complete an annual security questionnaire. Critical vendors are reviewed by the security team before contract renewal.

## 6.2 Business Continuity

Critical payment systems have a recovery time objective (RTO) of 24 hours. Disaster recovery drills are conducted once a year.
