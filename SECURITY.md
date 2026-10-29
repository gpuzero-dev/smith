# Security

## Reporting a vulnerability

Please do not open a public issue. Report it privately, either through
GitHub's [private vulnerability reporting](https://github.com/gpuzero-dev/smith/security/advisories/new)
or by email to vecchionejonathan@gmail.com.

Say what is affected, how to reproduce it, and what an attacker gains. You
will get an answer within a week, a fix or a plan for one, and credit in the
release notes if you want it.

## Supported versions

Only the latest release receives fixes. Smith is 0.x, and a fix
is released as a new version, never backported.

## What to know before you rely on it

- smith runs only the scripts its configuration lists, with parameters whose
  names and values it checks; it never runs a command it is sent.
- A node token authenticates it to its backend. Treat it like a password:
  give it in the environment (`SMITH_TOKEN`) rather than as `token` in a
  configuration file, and never commit a file that holds it.
- Its scripts run with the permissions of the user that starts smith.
