# SSH Upload Senza Password (Guida Generale)

Questa guida spiega come configurare SSH key-based authentication per evitare di inserire la password a ogni `ssh`, `scp` o `rsync`.

## Obiettivo

Dopo la configurazione, i comandi di upload verso server remoti useranno una chiave privata locale invece della password.

## Requisiti

- Accesso SSH funzionante al server (`username@host`)
- OpenSSH installato lato client
- Permessi di scrittura nella home remota (`~/.ssh/authorized_keys`)

---

## 1. Genera una chiave SSH (consigliata: ed25519)

### Windows (PowerShell)

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### macOS / Linux

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Premi Invio per usare il path di default.

Path tipici:
- Windows: `C:\Users\<user>\.ssh\id_ed25519`
- macOS/Linux: `~/.ssh/id_ed25519`

Consiglio: imposta una passphrase per maggiore sicurezza.

---

## 2. Avvia SSH agent e carica la chiave

### Windows (PowerShell)

```powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

Se `Set-Service` fallisce per permessi, apri PowerShell come Amministratore.

### macOS / Linux

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

## 3. Copia la chiave pubblica sul server

### Metodo A (piu semplice, se disponibile)

```bash
ssh-copy-id username@host
```

### Metodo B (manuale)

1. Mostra la chiave pubblica locale:

Windows:
```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

macOS/Linux:
```bash
cat ~/.ssh/id_ed25519.pub
```

2. Sul server remoto, crea la cartella `.ssh` se non esiste:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

3. Incolla la chiave in `~/.ssh/authorized_keys`:

```bash
nano ~/.ssh/authorized_keys
```

4. Imposta i permessi corretti:

```bash
chmod 600 ~/.ssh/authorized_keys
```

---

## 4. (Opzionale) Configura `~/.ssh/config`

Questo evita di riscrivere host, porta e username.

### Windows
File: `C:\Users\<user>\.ssh\config`

### macOS/Linux
File: `~/.ssh/config`

Esempio:

```sshconfig
Host mycluster
    HostName your.server.domain
    User your_username
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Poi puoi usare:

```bash
ssh mycluster
scp local_file mycluster:~/remote/path/
```

---

## 5. Verifica

```bash
ssh username@host "echo SSH_OK"
```

Se la configurazione e corretta, non verra richiesta la password dell'account remoto (potrebbe restare la passphrase della chiave, se non gestita dall'agent).

---

## 6. Troubleshooting rapido

### Errore: `Permission denied (publickey)`

- Verifica che la chiave pubblica sia in `~/.ssh/authorized_keys`
- Verifica i permessi:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

- Verifica che la chiave sia caricata:

```bash
ssh-add -l
```

### Errore su Windows: servizio `ssh-agent` non parte

- Apri PowerShell come Amministratore
- Riesegui:

```powershell
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
```

### Debug connessione

```bash
ssh -v username@host
```

Output utile: mostra quale chiave viene proposta e perche viene rifiutata.

---

## 7. Buone pratiche di sicurezza

- Usa sempre passphrase sulla chiave privata
- Non condividere mai la chiave privata (`id_ed25519`)
- Ruota la chiave se sospetti compromissione
- Rimuovi chiavi obsolete da `authorized_keys`
