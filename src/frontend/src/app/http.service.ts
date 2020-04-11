import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ICubeResponse, IActionRequest, IScrambleRequest, IScrambleResponse, IInfoResponse, ISolveRequest, ISolveResponse } from './rubiks/rubiks';

const urls = {
  info: "info",
  solved: "solved",
  action: "action",
  scramble: "scramble",
  solve: "solve",
}

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  hosts = [
    { name: "Heroku", address: "https://rl-rubiks.herokuapp.com" },
    { name: "Lokal", address: "http://127.0.0.1:5000" },
  ];
  selectedHost = this.hosts[0];

  constructor(private http: HttpClient) { }

  private getUrl(path: string) {
    return `${this.selectedHost.address}/${path}`;
  }

  public async getInfo() {
    return this.http.get<IInfoResponse>(this.getUrl(urls.info)).toPromise();
  }

  public async getSolved() {
    return this.http.get<ICubeResponse>(this.getUrl(urls.solved)).toPromise();
  }

  public async performAction(actionRequest: IActionRequest) {
    return this.http.post<ICubeResponse>(this.getUrl(urls.action), actionRequest).toPromise();
  }

  public async scramble(scrambleRequest: IScrambleRequest) {
    return this.http.post<IScrambleResponse>(this.getUrl(urls.scramble), scrambleRequest).toPromise();
  }

  public async solve(solveRequest: ISolveRequest) {
    return this.http.post<ISolveResponse>(this.getUrl(urls.solve), solveRequest).toPromise();
  }
}