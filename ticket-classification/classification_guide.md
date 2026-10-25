# IT-Support Ticket Classification Guide

| Field | Value |
| --- | --- |
| Version | 1.0 |
| Date | 2026-10-25 |
| Status | Draft, retrofitted from past classifications; not yet human-reviewed |
| Sources | `artifacts/datasets/train_dataset_v2.jsonl`, `validation_dataset_v2.jsonl`, `test_dataset.jsonl` |
| Specification | [019](../specs/019-ticket-classification-guide.md) |
| Derived system prompts | [prompts/](prompts/README.md) ([specification 020](../specs/020-ticket-classification-system-prompt.md)) |

This guide is the single reference for classifying IT-support tickets. The original rules used to generate the datasets are no longer available, so the guide was reconstructed from past classifications. Prompts, new training data, and label audits must follow it. Change it only through a new version, and record the reason in the change log at the end.

## 1. Output

Each ticket receives exactly one JSON object:

```json
{"category": "DB-02", "severity": "P2", "summary": "Finance database connection is failing"}
```

* `category`: exactly one code from section 2.
* `severity`: exactly one of `P1`, `P2`, `P3`, `P4`, from section 4.
* `summary`: see section 5.

## 2. Categories

Classify by the component that has failed, not by the symptom the user sees and not by the business process affected.

### APP-01 — Business application malfunction

An internal or customer-facing business application behaves incorrectly while the infrastructure under it works: errors, crashes, freezes, blank pages, wrong values or calculations, broken buttons, failing reports or exports.

Typical applications: HR, payroll, CRM, expenses, procurement, order entry, inventory, billing, case management, knowledge base, self-service and customer portals, reporting and analytics front ends.

### AUTH-01 — Authentication and account access

A legitimate user cannot prove their identity or use their account: login or sign-in failures, rejected passwords, expired or temporary credentials, locked or disabled accounts after the user's own failed attempts, MFA codes or approvals not arriving or rejected, tokens, single sign-on, session loops, identity service outages, VPN login rejected.

### DB-01 — Database performance

The database is reachable and accepts sessions, but it is slow: query latency, slow reports and dashboards, delayed transactions, batch or ETL jobs exceeding their window, reduced throughput, queuing.

### DB-02 — Database availability

The database, data store, listener, or endpoint is unavailable or refuses connections: connection timeouts, rejected sessions, lost connectivity between an application and its database, instances not responding, unavailable standby or backup databases.

### DEV-01 — End-user computer hardware

Physical faults of laptops, desktops, and workstations and of their peripherals: displays, monitors, docking stations, keyboards, mice, touchpads, headsets, built-in microphones and webcams, speakers, batteries, chargers, power supplies, USB ports, fans, local storage, physical damage.

### MAIL-01 — Email, calendar, and collaboration

Email, calendar, chat, and meeting services: mailboxes, sending and receiving, delays, attachments, mailbox search, shared mailboxes and calendars, invitations and scheduling, chat history, video meetings, and meeting-room conference audio and video equipment.

### MOB-01 — Corporate mobile devices

Company phones and tablets, including their hardware (screen, touch input, battery, charging port, camera, speaker, buttons, SIM, sensors, storage), boot and restart problems, and device-management enrollment and profiles.

### NET-01 — Network connectivity

LAN, Wi-Fi, internet access, VPN tunnels and remote access, DNS and name resolution, routing, proxy, IP addressing, packet loss, latency, and links between offices, branches, data centers, and cloud services.

### SEC-01 — Security incidents

Suspected or confirmed threats: phishing and impersonation messages, suspicious links or attachments, malware, unauthorized or suspicious logins and MFA prompts, unusual authentication activity, compromised accounts, communication with malicious destinations, data exposure or misdirected confidential data, security alerts, and policy violations.

### SRV-01 — Servers and compute infrastructure

Physical and virtual servers, hosts, VMs, and cluster nodes: operating system failures, boot failures, unresponsive hosts, CPU or memory exhaustion, resource contention, full or failing disks, volumes, and filesystems.

## 3. Boundary rules

Apply the first rule that matches.

| Situation | Category | Not |
| --- | --- | --- |
| Login fails and there are signs that someone other than the user is involved (login not initiated by the user, unknown location, unsolicited MFA prompts, phishing) | SEC-01 | AUTH-01 |
| Account disabled automatically because of suspicious activity | SEC-01 | AUTH-01 |
| Legitimate user cannot log in, including a lockout after their own failed attempts | AUTH-01 | SEC-01 |
| Authentication requests of an application or portal fail | AUTH-01 | APP-01 |
| The user logs in successfully, then the application fails | APP-01 | AUTH-01 |
| VPN credentials or VPN login rejected | AUTH-01 | NET-01 |
| VPN tunnel cannot be established or keeps disconnecting | NET-01 | AUTH-01 |
| Database reachable but slow | DB-01 | DB-02 |
| Database or its endpoint refuses or times out connections, including from one application | DB-02 | NET-01, APP-01 |
| Host, OS, CPU, memory, or disk of a server fails, even if it hosts a database | SRV-01 | DB-02 |
| The database rejects sessions while its server is healthy | DB-02 | SRV-01 |
| Phone or tablet, including its camera, speaker, or storage | MOB-01 | DEV-01 |
| Laptop, desktop, or workstation peripheral, including a headset microphone or built-in webcam not detected by the computer | DEV-01 | MAIL-01 |
| Meeting-room conference microphone or camera, or the meeting application not exposing a device | MAIL-01 | DEV-01 |
| A device has no network while the same port or network works for other devices | NET-01 | DEV-01 |
| End-user workstation | DEV-01 | SRV-01 |
| Application error with no identified infrastructure cause | APP-01 | SRV-01 |

## 4. Severity

### 4.1 Procedure

1. **Explicit impact first.** If the ticket explicitly states who is affected, how badly work is impacted, or whether a workaround exists, assign severity from those statements using the table in 4.2. The technical problem does not change the level. For example, an unavailable primary data store described as "low impact, can be handled during normal support work" is P4.
2. **Context otherwise.** If the ticket does not state its impact, infer it from the contextual signals in 4.3 to 4.6.
3. **Mixed signals.** When a stated dimension and an inferred dimension point to different levels, the stated one wins. When the stated dimensions are ambiguous or conflict, the stated business impact decides, then scope, then the workaround adjustment in 4.2. For example, "many users" with critical work stopped is P1, while "many users" with work significantly disrupted is P2.

### 4.2 Levels

| Level | Who is affected | Business impact | Workaround |
| --- | --- | --- | --- |
| P1 | Company-wide, most users, all production applications, or a main business or customer-facing service | Critical work or a critical business process is stopped | None |
| P2 | A large department, several teams, one office, building, or site, or one important production service or application | An important process is severely degraded or a team cannot work | No practical workaround |
| P3 | One user or a small group | Work is hindered, or one user is blocked | A workaround or alternative may exist |
| P4 | One user, or nobody yet | Minor, cosmetic, optional, or non-urgent; normal work continues | Easy workaround, or none needed |

Workaround adjustment: when the scope suggests P1 or P2 but the ticket states that a practical workaround is available, lower the level by one. Workarounds do not lower P3 or P4. When one user or a small group is fully blocked, the work depends on that resource, and there is no spare or alternative, raise P3 to P2.

### 4.3 Contextual signals

| Signal | Effect |
| --- | --- |
| "All", "every", "company-wide", "multiple regions", "all production applications" | P1 |
| A whole office or site completely disconnected, with no alternate path | P1 |
| Production or customer-facing service completely unavailable for many users or applications | P1 |
| Production critical workloads dropped with no redundancy | P1 |
| One department, team, office, building, wing, site, or one production application affected | P2 |
| A group of administrators or other key roles blocked | P2 |
| Production service severely slow or degraded, but still working | P2 |
| One server or VM down, unstable, or out of resources, when it is production or the ticket states service impact (timeouts, late jobs, failed writes) | P2 |
| One server or VM with a fault but no stated production role or service impact | P3 |
| The only device of a team or of a key role fails and there is no spare | P2 |
| One user, one device, one endpoint, one application feature, or a small group affected | P3 |
| Failure isolated to one component while others keep working | P3 |
| Cosmetic defect, rarely used or optional feature, single slow report or query | P4 |
| Device or service still usable with a minor limitation | P4 |
| Warning with no current service impact (for example low disk space) | P4 |

### 4.4 Security events (SEC-01)

| Situation | Severity |
| --- | --- |
| Active or likely compromise of production, privileged, or sensitive systems, malware on servers supporting critical applications, or suspicious activity across several production hosts | P1 |
| Evidence of compromise or exposure limited to one user, endpoint, account, or document: a successful login the user denies, MFA prompts the user did not initiate, repeated unsolicited authentication notifications, repeated login attempts from unusual locations, malware detected on one workstation, traffic to a known malicious destination, confidential data sent to the wrong recipient or exposed to unauthorized users | P2 |
| Suspicious event without evidence of compromise: a phishing or impersonation message, a suspicious attachment or link (even if opened), an account disabled automatically with no sign of wider compromise | P3 |
| Alert investigated with no confirmed compromise, or a policy violation with no active attack | P4 |

### 4.5 Non-production environments

Test, development, lab, staging, standby, and backup environments are not production.

* Unavailable or slow, with an alternative environment or work that can be postponed: P4.
* A team is blocked with no alternative environment, or redundancy of a production service is lost (for example a standby database is down while the primary works): P3.
* A non-production problem is never above P3 unless the ticket states a higher impact explicitly.

### 4.6 Outage versus degradation

* Complete outage of a production service: P1 when it affects many users or all applications, P2 when it affects one team or one application.
* Severe degradation of a production service: P1 only when it is stated as company-wide or unacceptable for everyone; otherwise P2.
* Degradation limited to one team, endpoint, application, or type of operation: P3.
* One slow report, query, or scheduled job that still completes: P4.

## 5. Summary

* English, one short noun phrase of 3 to 8 words.
* Name the component, then its state: "Payroll application is unavailable", "VPN tunnel cannot be established", "Suspicious login security incident".
* Preferred state words: `is failing`, `is malfunctioning`, `is unavailable`, `is unreachable`, `is degraded`, `is slow`, `is running slowly`, `is damaged`, `is missing`, `is locked`.
* Do not include user names, identifiers, locations, severity, or speculation about causes.

## 6. Examples

The examples are paraphrased and are not dataset records.

| Ticket | Category | Severity | Why |
| --- | --- | --- | --- |
| The CRM client crashes for one sales representative; there is no wider impact. | APP-01 | P3 | One user, application failure |
| A large department cannot complete MFA; there is no practical workaround. | AUTH-01 | P2 | Explicit: large department, no workaround |
| A reporting database is unavailable in the local setup; low impact, handled during normal support work. | DB-02 | P4 | Explicit impact wins over the technical problem |
| Every customer-facing service has lost its connection to the production database. | DB-02 | P1 | Production outage for all services |
| One dashboard refresh takes longer, but it still completes. | DB-01 | P4 | Single slow report |
| The only laptop of a small support group has a motherboard failure and the group cannot work. | DEV-01 | P2 | Only device of a team, no spare |
| A conference-room camera is not detected by the meeting system for one team. | MAIL-01 | P3 | Meeting-room equipment, small group |
| A warehouse tablet repeatedly reports storage errors and apps cannot save data. | MOB-01 | P3 | One device blocked |
| Employees in one building have very high Wi-Fi latency. | NET-01 | P2 | One building |
| An employee opened a link from a suspicious email; no sign of compromise yet. | SEC-01 | P3 | Suspicious event without evidence |
| A test VM is offline; engineers can use another environment. | SRV-01 | P4 | Non-production with alternative |

## 7. Known inconsistencies in past labels

These past labels are not explained by this guide. Review them during the label audit and either correct the label or change the guide.

* Test: a laptop whose fan runs continuously and that powers off during normal work is P2, while a desktop whose power supply has failed and will not start is P3. Both are one user's device; the guide gives P3.
* Test: a laptop whose local storage is failing, with files intermittently inaccessible, is P2. It is one user's device; the guide gives P3.
* Test: two VMs with exhausted memory (one unresponsive, one slow) are P2, although neither ticket states a production role or service impact; the guide gives P3.
* Test: an overnight data load that still completes but takes twice its normal window is P2; the guide treats one scheduled job that completes as P4.
* Training and validation: some generated descriptions are not physically meaningful (for example a phone speaker that "does not charge correctly", a conference microphone that "does not detect the microphone"). Their labels follow the guide, but the text should be rewritten.

## 8. Change log

* 1.0 (2026-10-25): First version, retrofitted from the training, validation, and test datasets.
