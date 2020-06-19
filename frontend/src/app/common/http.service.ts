import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { IInfoResponse, ISolveRequest, ISolveResponse, host } from './rubiks';

const urls = {
  info: "info",
  solve: "solve",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http: HttpClient) { }

  hosts: host[] = [
    { name: "Heroku", address: "https://rl-rubiks.herokuapp.com" },
    { name: "Local", address: "http://127.0.0.1:8000" },
  ];

  private _selectedHost = this.hosts[0];
  private _previousHost = this.selectedHost;

  get selectedHost() {
    return this._selectedHost;
  }

  set selectedHost(host: host) {
    this._previousHost = this._selectedHost;
    this._selectedHost = host;
  }

  get previousHost() {
    return this._previousHost;
  }

  private getUrl(path: string) {
    return `${this._selectedHost.address}/${path}`;
  }

  public async getInfo() {
    return this.http.get<IInfoResponse>(this.getUrl(urls.info)).toPromise();
  }

  public async solve(solveRequest: ISolveRequest) {
    return this.http.post<ISolveResponse>(this.getUrl(urls.solve), solveRequest).toPromise();
  }
}
